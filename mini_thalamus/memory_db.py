"""Transactional SQLite foundation for Mini Thalamus memory."""

from contextlib import contextmanager
import os
import sqlite3
import time


SCHEMA_VERSION = 1


class MemoryDatabase:
    """Own SQLite connection policy, schema initialization and transactions."""

    def __init__(self, db_path: str, busy_timeout_ms: int = 3000):
        self.db_path = os.path.abspath(db_path)
        self.busy_timeout_ms = int(busy_timeout_ms)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def initialize(self):
        with self.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            if row is not None and int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(
                    "memory database schema is newer than this program supports"
                )

            schema_sql = (
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    recent_messages_json TEXT NOT NULL DEFAULT '[]',
                    last_result_json TEXT,
                    agent_history_json TEXT NOT NULL DEFAULT '[]',
                    max_context_tokens INTEGER NOT NULL DEFAULT 8000,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    agent_id TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'running', 'success', 'error',
                            'checkpoint', 'interrupted'
                        )
                    ),
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    trace_id TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    UNIQUE (session_id, sequence_no)
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_status
                    ON turns(session_id, status, sequence_no);
                CREATE INDEX IF NOT EXISTS idx_turns_user
                    ON turns(user_id, created_at);

                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    gist TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    source_turn_id TEXT NOT NULL,
                    source_agent TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'active', 'superseded', 'rejected')
                    ),
                    version INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (source_session_id)
                        REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    FOREIGN KEY (source_turn_id)
                        REFERENCES turns(turn_id) ON DELETE RESTRICT,
                    UNIQUE (user_id, namespace, memory_key, version)
                );

                CREATE INDEX IF NOT EXISTS idx_ltm_user_status
                    ON long_term_memories(user_id, status, namespace, memory_key);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ltm_one_active
                    ON long_term_memories(user_id, namespace, memory_key)
                    WHERE status = 'active';

                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, state_key),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );
                """
            )
            for statement in schema_sql.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def recover_interrupted_turns(self, cutoff_ts: float) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE turns
                SET status='interrupted', completed_at=?
                WHERE status='running' AND created_at < ?
                """,
                (time.time(), float(cutoff_ts)),
            )
            return cursor.rowcount


__all__ = ["MemoryDatabase", "SCHEMA_VERSION"]
