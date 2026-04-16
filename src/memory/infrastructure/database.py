import sqlite3

from src.memory.infrastructure.utils import _ensure_parent_dir


class SQLiteRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        _ensure_parent_dir(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
