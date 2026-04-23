"""
HomeShield Database — SQLite event logging, camera config, zone storage.
"""
import sqlite3
import json
import threading
from datetime import datetime
from config import Config


class Database:
    _local = threading.local()

    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self._init_db()

    def _get_conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                url         TEXT NOT NULL,
                location    TEXT DEFAULT '',
                active      INTEGER DEFAULT 1,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS zones (
                zone_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name   TEXT NOT NULL,
                camera_id   INTEGER NOT NULL,
                polygon     TEXT NOT NULL,
                active      INTEGER DEFAULT 1,
                FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT NOT NULL,
                camera_id   INTEGER,
                camera_name TEXT,
                person_category TEXT DEFAULT 'unknown',
                confidence  REAL DEFAULT 0.0,
                snapshot_path TEXT DEFAULT '',
                alert_sent  INTEGER DEFAULT 0,
                details     TEXT DEFAULT '',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        conn.commit()

    # ── Cameras ──────────────────────────────────────────────
    def add_camera(self, name, url, location=""):
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO cameras (name, url, location) VALUES (?, ?, ?)",
            (name, url, location),
        )
        conn.commit()
        return cur.lastrowid

    def get_cameras(self, active_only=False):
        conn = self._get_conn()
        q = "SELECT * FROM cameras"
        if active_only:
            q += " WHERE active = 1"
        return [dict(r) for r in conn.execute(q).fetchall()]

    def update_camera(self, camera_id, **kwargs):
        conn = self._get_conn()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [camera_id]
        conn.execute(f"UPDATE cameras SET {sets} WHERE camera_id = ?", vals)
        conn.commit()

    def delete_camera(self, camera_id):
        conn = self._get_conn()
        conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
        conn.execute("DELETE FROM zones WHERE camera_id = ?", (camera_id,))
        conn.commit()

    # ── Zones ────────────────────────────────────────────────
    def add_zone(self, zone_name, camera_id, polygon):
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO zones (zone_name, camera_id, polygon) VALUES (?, ?, ?)",
            (zone_name, camera_id, json.dumps(polygon)),
        )
        conn.commit()
        return cur.lastrowid

    def get_zones(self, camera_id=None):
        conn = self._get_conn()
        if camera_id:
            rows = conn.execute(
                "SELECT * FROM zones WHERE camera_id = ? AND active = 1", (camera_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM zones WHERE active = 1").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["polygon"] = json.loads(d["polygon"])
            result.append(d)
        return result

    def delete_zone(self, zone_id):
        conn = self._get_conn()
        conn.execute("DELETE FROM zones WHERE zone_id = ?", (zone_id,))
        conn.commit()

    # ── Events ───────────────────────────────────────────────
    def log_event(self, event_type, camera_id=None, camera_name="", person_category="unknown",
                  confidence=0.0, snapshot_path="", alert_sent=False, details=""):
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO events
               (event_type, camera_id, camera_name, person_category, confidence,
                snapshot_path, alert_sent, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_type, camera_id, camera_name, person_category,
             round(confidence, 3), snapshot_path, int(alert_sent), details),
        )
        conn.commit()
        return cur.lastrowid

    def get_events(self, limit=100, event_type=None, camera_id=None):
        conn = self._get_conn()
        q = "SELECT * FROM events"
        params = []
        clauses = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(q, params).fetchall()]

    def clear_events(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM events")
        conn.commit()

    def get_today_alert_count(self):
        conn = self._get_conn()
        today = datetime.now().strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type != 'normal' AND date(created_at) = ?",
            (today,),
        ).fetchone()
        return row["cnt"] if row else 0

    # ── Settings ─────────────────────────────────────────────
    def get_setting(self, key, default=None):
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        conn.commit()