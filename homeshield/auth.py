"""User accounts, password hashing, and Flask session helpers.

Two roles:
  * admin  -- full access to all routes, including user management
  * guest  -- read-only access to live feeds and the event log

On first start (no users in the DB) we seed a default `admin / admin`
account with the must-change-password flag set, so the first login is
forced to pick a real password before reaching the dashboard.
"""

from __future__ import annotations

import sqlite3
import threading
from functools import wraps
from typing import Any, Optional

from flask import jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash

from .db import read_conn, write_conn


ROLE_ADMIN = "admin"
ROLE_GUEST = "guest"
VALID_ROLES = (ROLE_ADMIN, ROLE_GUEST)

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"

MIN_PASSWORD_LENGTH = 4   # keep low so demo / FYP scenarios stay friendly


class UserStore:
    """SQLite-backed user store. Thread-safe by virtue of write_conn()."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_default_admin()

    # ---- bootstrap ------------------------------------------------------

    def _ensure_default_admin(self) -> None:
        with read_conn(self.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        if n == 0:
            self.create_user(
                username=DEFAULT_ADMIN_USERNAME,
                password=DEFAULT_ADMIN_PASSWORD,
                role=ROLE_ADMIN,
                must_change=True,
            )
            print(f"[auth] seeded default admin "
                  f"({DEFAULT_ADMIN_USERNAME}/{DEFAULT_ADMIN_PASSWORD}) - "
                  f"you will be forced to set a new password on first login")

    # ---- CRUD -----------------------------------------------------------

    def list_users(self) -> list[dict[str, Any]]:
        with read_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT user_id, username, role, must_change, created_at "
                "FROM users ORDER BY user_id ASC"
            ).fetchall()
        return [self._row_to_public(r) for r in rows]

    def get(self, user_id: int) -> Optional[dict[str, Any]]:
        with read_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        return dict(r) if r else None

    def get_by_username(self, username: str) -> Optional[dict[str, Any]]:
        username = (username or "").strip()
        if not username:
            return None
        with read_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,),
            ).fetchone()
        return dict(r) if r else None

    def create_user(self, *, username: str, password: str,
                    role: str = ROLE_GUEST,
                    must_change: bool = False) -> dict[str, Any]:
        username = (username or "").strip()
        if not username:
            raise ValueError("username is required")
        if len(username) > 64:
            raise ValueError("username is too long")
        if not password or len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"password must be at least {MIN_PASSWORD_LENGTH} characters"
            )
        if role not in VALID_ROLES:
            role = ROLE_GUEST
        pw_hash = generate_password_hash(password)
        try:
            with write_conn(self.db_path) as conn:
                cur = conn.execute(
                    """INSERT INTO users (username, password_hash, role, must_change)
                       VALUES (?, ?, ?, ?)""",
                    (username, pw_hash, role, 1 if must_change else 0),
                )
                uid = cur.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"username '{username}' already exists")
        return {
            "user_id": uid,
            "username": username,
            "role": role,
            "must_change_password": bool(must_change),
        }

    def delete_user(self, user_id: int) -> bool:
        """Refuses to delete the last admin so the system stays manageable."""
        user_id = int(user_id)
        target = self.get(user_id)
        if target is None:
            return False
        if target["role"] == ROLE_ADMIN and self.count_admins() <= 1:
            raise ValueError("cannot delete the last admin")
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM users WHERE user_id = ?", (user_id,)
            )
            return (cur.rowcount or 0) > 0

    def update_role(self, user_id: int, role: str) -> bool:
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role: {role}")
        user_id = int(user_id)
        target = self.get(user_id)
        if target is None:
            return False
        # Demoting the last admin would lock everyone out of /api/users.
        if target["role"] == ROLE_ADMIN and role != ROLE_ADMIN \
                and self.count_admins() <= 1:
            raise ValueError("cannot demote the last admin")
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE users SET role = ? WHERE user_id = ?",
                (role, user_id),
            )
            return (cur.rowcount or 0) > 0

    def update_password(self, user_id: int, new_password: str) -> bool:
        if not new_password or len(new_password) < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"password must be at least {MIN_PASSWORD_LENGTH} characters"
            )
        pw_hash = generate_password_hash(new_password)
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ?, must_change = 0 "
                "WHERE user_id = ?",
                (pw_hash, int(user_id)),
            )
            return (cur.rowcount or 0) > 0

    def count_admins(self) -> int:
        with read_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE role = ?", (ROLE_ADMIN,),
            ).fetchone()
        return int(r["c"])

    # ---- authentication -------------------------------------------------

    def verify(self, username: str, password: str) -> Optional[dict[str, Any]]:
        row = self.get_by_username(username)
        if not row:
            return None
        if not check_password_hash(row["password_hash"], password):
            return None
        return row

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _row_to_public(r) -> dict[str, Any]:
        return {
            "user_id": r["user_id"],
            "username": r["username"],
            "role": r["role"],
            "must_change_password": bool(r["must_change"]),
            "created_at": r["created_at"],
        }


# ---- session helpers ------------------------------------------------------

def session_user_id() -> Optional[int]:
    uid = session.get("user_id")
    try:
        return int(uid) if uid is not None else None
    except (TypeError, ValueError):
        return None


def session_role() -> Optional[str]:
    return session.get("role")


def session_must_change() -> bool:
    return bool(session.get("must_change"))


def login_session(user_row: dict[str, Any]) -> None:
    session.clear()
    session["user_id"] = int(user_row["user_id"])
    session["username"] = user_row["username"]
    session["role"] = user_row["role"]
    session["must_change"] = bool(user_row.get("must_change", 0))
    session.permanent = True


def logout_session() -> None:
    session.clear()


# ---- decorators -----------------------------------------------------------

def require_login(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if session_user_id() is None:
            return jsonify({"error": "auth_required"}), 401
        if session_must_change():
            # The only thing a must-change user is allowed to do is set a
            # new password. Treat everything else as auth-required so the
            # frontend re-renders the change-password overlay.
            return jsonify({"error": "password_change_required"}), 401
        return view(*args, **kwargs)
    return wrapper


def require_admin(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if session_user_id() is None:
            return jsonify({"error": "auth_required"}), 401
        if session_must_change():
            return jsonify({"error": "password_change_required"}), 401
        if session_role() != ROLE_ADMIN:
            return jsonify({"error": "admin_required"}), 403
        return view(*args, **kwargs)
    return wrapper
