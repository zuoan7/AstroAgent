"""SQLite 仓储基类。

短期记忆各 repository 共用这里的连接参数、row_factory 和 WAL 配置。
"""

import sqlite3

from src.memory.infrastructure.utils import _ensure_parent_dir


class SQLiteRepository:
    """提供记忆模块 SQLite 连接的基础类。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        _ensure_parent_dir(db_path)

    def _connect(self) -> sqlite3.Connection:
        """创建启用 WAL、外键和 Row 访问模式的 SQLite 连接。"""

        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
