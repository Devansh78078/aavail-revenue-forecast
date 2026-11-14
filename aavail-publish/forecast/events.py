"""Store model and request events in a concurrent SQLite journal."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS events_kind_id ON events (kind, id)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def write(self, kind, **payload):
        with self.connection() as db:
            db.execute(
                "INSERT INTO events (timestamp, kind, payload) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), kind, json.dumps(payload, allow_nan=False)),
            )

    def read(self, kind=None, limit=None):
        query = "SELECT * FROM events"
        values = []
        if kind is not None:
            query += " WHERE kind = ?"
            values.append(kind)
        query += " ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)
        with self.connection() as db:
            return [
                dict(id=row["id"], timestamp=row["timestamp"], kind=row["kind"], **json.loads(row["payload"]))
                for row in db.execute(query, values)
            ]
