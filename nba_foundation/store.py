"""Append-only SQLite snapshots with conservative system-knowledge cutoffs.

Each source/feed/scope is a complete replacement snapshot, not a record delta.
This prevents deleted or corrected events from reappearing through row-level joins.
"""

from contextlib import AbstractContextManager
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Callable

from .timeutil import now, utc

ENTITIES = frozenset({
    "games", "players", "roster_history", "availability_reports", "events", "shots",
    "team_box_scores", "player_box_scores", "id_mappings", "possessions",
    "lineup_stints", "action_tags", "tracking_frames", "prediction_snapshots",
})
STATUSES = frozenset({"observed", "derived", "estimated", "simulated", "unavailable"})
DATA_CLASSES = frozenset({"real", "test", "scrambled", "unverified"})
PARSER_VERSION = "1"


class DataUnavailable(ValueError):
    """Inputs are absent, quarantined, stale, or not real at the requested cutoff."""


class Archive(AbstractContextManager):
    def __init__(self, path: str | Path, *, clock: Callable[[], str] = now):
        self.clock = clock
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.db.close()
            raise ValueError(f"Unsupported archive schema: {version}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL, feed TEXT NOT NULL, scope TEXT NOT NULL,
                retrieved_at TEXT NOT NULL, source_published_at TEXT,
                available_at TEXT NOT NULL, data_class TEXT NOT NULL,
                parser_version TEXT NOT NULL, sha256 TEXT NOT NULL,
                raw_payload BLOB NOT NULL, normalization_error TEXT,
                source_uri TEXT, response_headers_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
                entity TEXT NOT NULL, record_key TEXT NOT NULL,
                status TEXT NOT NULL, event_time TEXT, source_updated_at TEXT,
                data_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, entity, record_key)
            );
            CREATE INDEX IF NOT EXISTS snapshots_lookup
                ON snapshots(source, feed, scope, retrieved_at, id);
            CREATE INDEX IF NOT EXISTS records_entity ON records(entity, snapshot_id);
            CREATE TRIGGER IF NOT EXISTS immutable_snapshots_update BEFORE UPDATE ON snapshots
                BEGIN SELECT RAISE(ABORT, 'Snapshots are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_snapshots_delete BEFORE DELETE ON snapshots
                BEGIN SELECT RAISE(ABORT, 'Snapshots are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_records_update BEFORE UPDATE ON records
                BEGIN SELECT RAISE(ABORT, 'Records are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_records_delete BEFORE DELETE ON records
                BEGIN SELECT RAISE(ABORT, 'Records are immutable'); END;
            PRAGMA user_version = 1;
        """)
        for entity in sorted(ENTITIES):
            self.db.execute(f"CREATE VIEW IF NOT EXISTS {entity} AS SELECT * FROM records WHERE entity = '{entity}'")

    def __exit__(self, *args):
        self.db.close()

    def ingest(self, raw: bytes, *, source: str, feed: str, scope: str,
               data_class: str = "unverified", source_published_at: str | None = None,
               source_uri: str | None = None, response_headers: dict | None = None) -> int:
        """Archive exact bytes even when normalization fails. No backdated ingestion.

        `clock` injection is for deterministic tests only. A historical source date
        cannot make a newly collected document visible to an earlier forecast.
        """
        from .adapters import normalize

        if not all(isinstance(value, str) and value.strip() for value in (source, feed, scope)):
            raise ValueError("source, feed and scope are required")
        if data_class not in DATA_CLASSES:
            raise ValueError("Invalid data classification")
        if not isinstance(raw, bytes):
            raise ValueError("Raw payload must be bytes")
        received = utc(self.clock())
        published = utc(source_published_at) if source_published_at else None
        available = max(received, published or received)
        error, rows = None, []
        try:
            payload = json.loads(raw)
            rows = normalize(feed, scope, payload)
            keys = set()
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Each record must be an object")
                if row["entity"] not in ENTITIES or row["status"] not in STATUSES:
                    raise ValueError("Unknown entity or data status")
                if not isinstance(row["key"], str) or not row["key"]:
                    raise ValueError("Record key must be a non-empty string")
                key = (row["entity"], row["key"])
                if key in keys:
                    raise ValueError(f"Duplicate record: {key}")
                keys.add(key)
                for field in ("event_time", "source_updated_at"):
                    row[field] = utc(row[field]) if row.get(field) else None
                if row["source_updated_at"]:
                    available = max(available, row["source_updated_at"])
                if not isinstance(row["data"], dict):
                    raise ValueError("Record data must be an object")
                row["encoded"] = json.dumps(row["data"], sort_keys=True, allow_nan=False)
        except (ValueError, TypeError, KeyError, OverflowError, AttributeError, IndexError) as exc:
            error, rows = f"{type(exc).__name__}: {exc}", []
        with self.db:
            cursor = self.db.execute("""
                INSERT INTO snapshots(source, feed, scope, retrieved_at, source_published_at,
                    available_at, data_class, parser_version, sha256, raw_payload, normalization_error,
                    source_uri, response_headers_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (source, feed, scope, received, published, available, data_class,
                  PARSER_VERSION, hashlib.sha256(raw).hexdigest(), raw, error,
                  source_uri, json.dumps(response_headers or {}, sort_keys=True)))
            snapshot_id = cursor.lastrowid
            self.db.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", [
                (snapshot_id, row["entity"], row["key"], row["status"], row["event_time"],
                 row["source_updated_at"], row["encoded"]) for row in rows
            ])
        return snapshot_id

    def metadata(self, snapshot_id: int) -> dict:
        row = self.db.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if row is None:
            raise DataUnavailable(f"Unknown snapshot {snapshot_id}")
        return {key: row[key] for key in row.keys() if key != "raw_payload"}

    def raw(self, snapshot_id: int) -> bytes:
        row = self.db.execute("SELECT raw_payload, sha256 FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if row is None:
            raise DataUnavailable(f"Unknown snapshot {snapshot_id}")
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise DataUnavailable("Raw payload hash mismatch")
        return row[0]

    def snapshot_as_of(self, *, source: str, feed: str, scope: str, cutoff: str,
                       max_age_seconds: float | None = None) -> dict:
        cutoff = utc(cutoff)
        # Select first, then validate. A broken/new test feed never silently falls
        # back to an older good snapshot. A future publication is not yet visible.
        row = self.db.execute("""
            SELECT id FROM snapshots WHERE source = ? AND feed = ? AND scope = ?
                AND retrieved_at <= ? AND available_at <= ?
            ORDER BY retrieved_at DESC, id DESC LIMIT 1
        """, (source, feed, scope, cutoff, cutoff)).fetchone()
        if row is None:
            raise DataUnavailable("No snapshot was known at this cutoff")
        result = self.metadata(row[0])
        if result["normalization_error"]:
            raise DataUnavailable(f"Snapshot {row[0]} quarantined: {result['normalization_error']}")
        if result["data_class"] != "real":
            raise DataUnavailable(f"Snapshot {row[0]} is {result['data_class']}, not verified real-source data")
        if max_age_seconds is not None:
            if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
                raise ValueError("max_age_seconds must be finite and nonnegative")
            age = (datetime.fromisoformat(cutoff) - datetime.fromisoformat(result["retrieved_at"])).total_seconds()
            if age > max_age_seconds:
                raise DataUnavailable(f"Snapshot {row[0]} is stale ({age:.0f}s old)")
        self.raw(row[0])
        return result

    def records_as_of(self, *, entity: str | None = None, **query) -> dict:
        if entity is not None and entity not in ENTITIES:
            raise ValueError("Unknown entity")
        snapshot = self.snapshot_as_of(**query)
        rows = self.db.execute("SELECT * FROM records WHERE snapshot_id = ? ORDER BY entity, record_key", (snapshot["id"],))
        return {"snapshot": snapshot, "records": [
            {"entity": row["entity"], "key": row["record_key"], "status": row["status"],
             "event_time": row["event_time"], "source_updated_at": row["source_updated_at"],
             "data": json.loads(row["data_json"])}
            for row in rows if entity is None or row["entity"] == entity
        ]}
