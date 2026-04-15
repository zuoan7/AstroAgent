import os
import time
from datetime import datetime
from typing import List, Optional

from src.core.logger import logger
from src.memory.long_term_memory.models import EventLogEntry, EventType, _utcnow_iso
from src.memory.long_term_memory.repository import LongTermMemoryRepository


class BackupManager:
    BACKUP_DIR_NAME = "backups"
    MAX_BACKUPS = 10

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        backup_dir: Optional[str] = None,
        max_backups: int = 10,
        auto_backup_interval_hours: int = 24,
    ):
        self._repo = repository
        self.max_backups = max_backups
        self.auto_backup_interval_hours = auto_backup_interval_hours

        if backup_dir:
            self.backup_dir = backup_dir
        else:
            db_dir = os.path.dirname(repository.db_path)
            self.backup_dir = os.path.join(db_dir, self.BACKUP_DIR_NAME)

        os.makedirs(self.backup_dir, exist_ok=True)
        self._last_backup_time = 0.0

    def _generate_backup_path(self, tag: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag_suffix = f"_{tag}" if tag else ""
        filename = f"ltm_backup_{timestamp}{tag_suffix}.sqlite"
        return os.path.join(self.backup_dir, filename)

    def create_backup(self, tag: Optional[str] = None) -> Optional[str]:
        backup_path = self._generate_backup_path(tag)
        success = self._repo.backup_database(backup_path)
        if success:
            self._last_backup_time = time.time()
            self._repo.add_event_log(EventLogEntry(
                user_id="__system__",
                event_type=EventType.BACKUP_CREATED,
                event_detail=f"数据库备份: {os.path.basename(backup_path)}",
                metadata={"backup_path": backup_path, "tag": tag or ""},
            ))
            self._cleanup_old_backups()
            logger.info(f"长期记忆备份创建成功: {backup_path}")
            return backup_path
        return None

    def restore_from_backup(self, backup_path: str) -> bool:
        if not os.path.exists(backup_path):
            logger.error(f"备份文件不存在: {backup_path}")
            return False

        pre_backup = self.create_backup(tag="pre_restore")
        if pre_backup:
            logger.info(f"恢复前创建安全备份: {pre_backup}")

        success = self._repo.restore_from_backup(backup_path)
        if success:
            self._repo.add_event_log(EventLogEntry(
                user_id="__system__",
                event_type=EventType.BACKUP_RESTORED,
                event_detail=f"数据库恢复: {os.path.basename(backup_path)}",
                metadata={"backup_path": backup_path, "pre_restore_backup": pre_backup or ""},
            ))
            logger.info(f"长期记忆恢复成功: {backup_path}")
        return success

    def list_backups(self) -> List[dict]:
        backups = []
        if not os.path.exists(self.backup_dir):
            return backups

        for filename in sorted(os.listdir(self.backup_dir), reverse=True):
            if filename.startswith("ltm_backup_") and filename.endswith(".sqlite"):
                filepath = os.path.join(self.backup_dir, filename)
                try:
                    stat = os.stat(filepath)
                    backups.append({
                        "filename": filename,
                        "path": filepath,
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except OSError:
                    continue
        return backups

    def delete_backup(self, backup_path: str) -> bool:
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
                logger.info(f"备份已删除: {backup_path}")
                return True
            return False
        except OSError as e:
            logger.error(f"删除备份失败: {e}")
            return False

    def _cleanup_old_backups(self):
        backups = self.list_backups()
        while len(backups) > self.max_backups:
            oldest = backups.pop()
            self.delete_backup(oldest["path"])

    def maybe_auto_backup(self) -> Optional[str]:
        now = time.time()
        interval_seconds = self.auto_backup_interval_hours * 3600
        if now - self._last_backup_time >= interval_seconds:
            return self.create_backup(tag="auto")
        return None
