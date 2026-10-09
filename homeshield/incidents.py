"""Incidents: alerts grouped into episodes someone can act on.

The detectors emit an event on every cooldown tick (a fire in view logs one
every 5 s), so the raw log reads as noise and nothing in it says whether
anyone dealt with an alert. An incident is one camera + one kind of trouble,
kept going while its events keep arriving less than INCIDENT_GAP_S apart.
It carries the worst stage reached, the peak confidence and its snapshot,
and how it was closed: acknowledged or false alarm, by whom, with a note.

Grouping happens on the server, inside the same transaction that stores the
event, so every phone and laptop in the house sees the same incidents.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, Optional

from .db import read_conn, write_conn

log = logging.getLogger(__name__)

INCIDENT_GAP_S = 120.0
NOTE_MAX = 200
OUTCOMES = ("acknowledged", "false_alarm")
STATUSES = ("open",) + OUTCOMES
CRITICAL, ATTENTION = 2, 1

# event type -> (kind, severity, rank). Rank orders the stages inside one
# kind, so a fall that turns into "lying motionless" is headlined by the
# later, worse stage. "normal" and "system" events never form incidents.
EVENT_KINDS: dict[str, tuple[str, int, int]] = {
    "fall_detected":     ("fall", ATTENTION, 1),
    "inactivity":        ("fall", ATTENTION, 2),
    "lying_motionless":  ("fall", CRITICAL, 3),
    "fire_detected":     ("fire", CRITICAL, 1),
    "intruder_detected": ("intruder", CRITICAL, 1),
    "zone_entry":        ("zone", ATTENTION, 1),
}
KINDS = ("fall", "fire", "intruder", "zone")

# Incidents still open when incidents were introduced are closed by the
# upgrade if they ended more than this long ago; the old log had no way to
# mark anything handled, so it would otherwise arrive as a wall of alarms.
BACKFILL_CLOSE_AFTER_S = 3600.0
UPGRADE_USER = "HomeShield (upgrade)"


def to_json(row, now: Optional[float] = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    d = dict(row)
    d["severity"] = "critical" if d["severity"] >= CRITICAL else "attention"
    d["escalated"] = bool(d["escalated"])
    d["active"] = (now - d["last_ts"]) < INCIDENT_GAP_S
    d["duration_s"] = max(0.0, d["last_ts"] - d["started_ts"])
    d.pop("headline_rank", None)
    return d


def assign(conn, ev) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """File a stored event into its incident; return (change, incident).

    change is "opened", "updated" or "escalated" (severity went up, which
    also reopens an incident that had been closed). Events that cannot form
    incidents return (None, None).
    """
    spec = EVENT_KINDS.get(ev.event_type)
    if spec is None:
        return None, None
    kind, sev, rank = spec
    conf = float(ev.confidence or 0.0)
    category = ev.person_category or "unknown"
    # The incident whose time span this event falls within the gap of. Events
    # normally arrive in order, but a late or backfilled one must not join an
    # incident from days later just because that one is the newest.
    row = conn.execute(
        "SELECT * FROM incidents WHERE kind = ? AND camera_id IS ? "
        "AND last_ts >= ? AND started_ts <= ? ORDER BY last_ts DESC LIMIT 1",
        (kind, ev.camera_id, ev.ts - INCIDENT_GAP_S, ev.ts + INCIDENT_GAP_S),
    ).fetchone()

    if row is None:
        cur = conn.execute(
            """INSERT INTO incidents
               (kind, camera_id, camera_name, headline_type, headline_rank,
                severity, started_ts, last_ts, event_count, peak_confidence,
                peak_event_id, snapshot_path, person_category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (kind, ev.camera_id, ev.camera_name, ev.event_type, rank, sev,
             ev.ts, ev.ts, conf, ev.event_id, ev.snapshot_path, category),
        )
        iid, change = cur.lastrowid, "opened"
    else:
        iid = row["incident_id"]
        sets: dict[str, Any] = {
            "started_ts": min(row["started_ts"], ev.ts),
            "last_ts": max(row["last_ts"], ev.ts),
            "event_count": row["event_count"] + 1,
            "camera_name": ev.camera_name or row["camera_name"],
        }
        if conf > row["peak_confidence"]:
            sets.update(peak_confidence=conf, peak_event_id=ev.event_id)
            if ev.snapshot_path:
                sets["snapshot_path"] = ev.snapshot_path
        elif not row["snapshot_path"] and ev.snapshot_path:
            sets["snapshot_path"] = ev.snapshot_path
        if rank > row["headline_rank"]:
            sets.update(headline_type=ev.event_type, headline_rank=rank)
        if category != "unknown":
            sets["person_category"] = category
        change = "updated"
        if sev > row["severity"]:
            # Worse than anything seen so far: always resurfaces, even if
            # someone already closed it. The old resolution stays on record
            # so the sheet can say who had acknowledged it.
            sets.update(severity=sev, escalated=1, status="open")
            change = "escalated"
        cols = ", ".join(f"{k} = ?" for k in sets)
        conn.execute(f"UPDATE incidents SET {cols} WHERE incident_id = ?",
                     (*sets.values(), iid))

    if ev.event_id is not None:
        conn.execute("UPDATE events SET incident_id = ? WHERE event_id = ?",
                     (iid, ev.event_id))
    inc = conn.execute("SELECT * FROM incidents WHERE incident_id = ?",
                       (iid,)).fetchone()
    return change, to_json(inc)


class IncidentStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    # ---- reads ------------------------------------------------------------

    def get(self, incident_id: int, *, with_events: bool = False,
            event_limit: int = 500) -> Optional[dict[str, Any]]:
        with read_conn(self.db_path) as conn:
            row = conn.execute("SELECT * FROM incidents WHERE incident_id = ?",
                               (int(incident_id),)).fetchone()
            if row is None:
                return None
            out = to_json(row)
            if with_events:
                evs = conn.execute(
                    "SELECT event_id, ts, event_type, confidence, details, "
                    "snapshot_path FROM events WHERE incident_id = ? "
                    "ORDER BY ts ASC LIMIT ?",
                    (int(incident_id), int(event_limit)),
                ).fetchall()
                out["events"] = [dict(e) for e in evs]
        return out

    def list(self, *, status: Optional[str] = None, kind: Optional[str] = None,
             camera_id: Optional[int] = None, limit: int = 50
             ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if status == "resolved":
            clauses.append("status != 'open'")
        elif status:
            clauses.append("status = ?")
            params.append(status)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(int(camera_id))
        sql = "SELECT * FROM incidents"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Open ones lead with the most serious; history reads newest first.
        sql += (" ORDER BY severity DESC, last_ts DESC" if status == "open"
                else " ORDER BY last_ts DESC")
        sql += " LIMIT ?"
        params.append(int(limit))
        now = time.time()
        with read_conn(self.db_path) as conn:
            return [to_json(r, now) for r in conn.execute(sql, params)]

    def board(self, *, open_limit: int = 20, recent: int = 6) -> dict[str, Any]:
        """What the Live board needs in one request: open first, then recent."""
        now = time.time()
        with read_conn(self.db_path) as conn:
            open_rows = conn.execute(
                "SELECT * FROM incidents WHERE status = 'open' "
                "ORDER BY severity DESC, last_ts DESC LIMIT ?",
                (int(open_limit),)).fetchall()
            totals = conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(status = 'open'), 0) AS open_n, "
                "COALESCE(SUM(status = 'open' AND severity >= ?), 0) AS crit_n "
                "FROM incidents", (CRITICAL,)).fetchone()
            recent_rows = conn.execute(
                "SELECT * FROM incidents WHERE status != 'open' "
                "ORDER BY last_ts DESC LIMIT ?", (int(recent),)).fetchall()
        return {
            "open": [to_json(r, now) for r in open_rows],
            "open_total": int(totals["open_n"]),
            "open_critical": int(totals["crit_n"]),
            "recent": [to_json(r, now) for r in recent_rows],
            "has_any": int(totals["n"]) > 0,
        }

    def open_counts(self) -> dict[str, int]:
        with read_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(severity >= ?), 0) AS c "
                "FROM incidents WHERE status = 'open'", (CRITICAL,)).fetchone()
        return {"open_incidents": int(r["n"]), "open_critical": int(r["c"])}

    def stats(self, *, days: int = 7) -> dict[str, Any]:
        """Outcome counts per detector and per camera: the false-alarm record."""
        since = time.time() - days * 86400
        q = ("SELECT {col} AS key, COUNT(*) AS total, "
             "COALESCE(SUM(status = 'open'), 0) AS open, "
             "COALESCE(SUM(status = 'acknowledged'), 0) AS acknowledged, "
             "COALESCE(SUM(status = 'false_alarm'), 0) AS false_alarm "
             "FROM incidents WHERE started_ts >= ? GROUP BY {col} "
             "ORDER BY total DESC")
        with read_conn(self.db_path) as conn:
            by_kind = [dict(r) for r in conn.execute(q.format(col="kind"), (since,))]
            by_camera = [dict(r) for r in conn.execute(
                q.format(col="COALESCE(camera_name, 'Unknown')"), (since,))]
        return {"days": days, "by_kind": by_kind, "by_camera": by_camera}

    # ---- writes -----------------------------------------------------------

    def resolve(self, incident_id: int, *, outcome: str, by: str,
                note: str = "") -> Optional[dict[str, Any]]:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE incidents SET status = ?, resolved_by = ?, "
                "resolved_at = ?, note = ? WHERE incident_id = ?",
                (outcome, by, time.time(), note.strip() or None, int(incident_id)))
            if cur.rowcount == 0:
                return None
        return self.get(incident_id)

    def backfill(self) -> int:
        """Group events logged before incidents existed. Runs once per event.

        Returns the number of incidents created.
        """
        types = tuple(EVENT_KINDS)
        marks = ",".join("?" * len(types))
        created: list[int] = []
        with write_conn(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT event_id, ts, event_type, camera_id, camera_name, "
                f"person_category, confidence, snapshot_path FROM events "
                f"WHERE incident_id IS NULL AND event_type IN ({marks}) "
                f"ORDER BY ts ASC", types).fetchall()
            for r in rows:
                change, inc = assign(conn, SimpleNamespace(**dict(r)))
                if change == "opened":
                    created.append(inc["incident_id"])
            if created:
                cutoff = time.time() - BACKFILL_CLOSE_AFTER_S
                marks_c = ",".join("?" * len(created))
                conn.execute(
                    f"UPDATE incidents SET status = 'acknowledged', "
                    f"resolved_by = ?, resolved_at = ?, note = ? "
                    f"WHERE incident_id IN ({marks_c}) AND status = 'open' "
                    f"AND last_ts < ?",
                    (UPGRADE_USER, time.time(),
                     "Grouped from the event log when incidents were added",
                     *created, cutoff))
        if created:
            log.info("grouped %d logged events into %d incidents",
                     len(rows), len(created))
        return len(created)
