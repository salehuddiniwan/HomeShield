"""Flask app implementing the HomeShield dashboard API."""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import re
import queue
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any

from flask import (Flask, Response, abort, jsonify, make_response,
                   render_template, request, send_from_directory, session)

from .auth import (REMEMBER_DAYS, ROLE_ADMIN, ROLE_GUEST, LoginThrottle,
                   UserStore, login_session, logout_session, require_admin,
                   require_login, session_user_id)
from .cameras import CameraManager, _strip_quotes
from .db import init_db, write_conn
from .detectors.face import is_good_face
from .events import Event, EventBus
from .incidents import KINDS, NOTE_MAX, OUTCOMES, STATUSES, IncidentStore
from .notify import WhatsAppNotifier
from .persons import IntruderStore, PersonStore
from .pipeline import Models, list_fire_models, list_pose_models
from .settings import SettingsStore
from .zones import ZoneStore

log = logging.getLogger(__name__)

# Text shrinks 4-5x under gzip; images and video are compressed already.
_GZIP_TYPES = frozenset({"text/html", "text/css", "text/plain", "text/javascript",
                         "application/javascript", "application/json",
                         "image/svg+xml"})
_GZIP_MIN_BYTES = 1024


# ---- settings POST coercion ----------------------------------------------

_INT_KEYS = ("inactivity_seconds", "alert_cooldown", "yolo_imgsz",
             "process_fps", "fire_cooldown", "intruder_cooldown",
             "fire_every_n", "face_every_n", "fire_confirm_frames",
             "fire_confirm_window", "face_min_size", "intruder_confirm_frames")
_FLOAT_KEYS = ("fall_threshold", "yolo_confidence", "fire_confidence",
               "face_match_threshold", "face_min_det_score")
_BOOL_KEYS = ("use_fp16", "fall_enabled", "fire_enabled", "face_enabled")


def _to_bool(v: Any) -> bool:
    # bool("false") is True, so handle the string spellings explicitly.
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _coerce_settings(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce typed settings; return (data, keys that could not be coerced).

    Invalid numbers must be rejected rather than stored as strings: the
    pipeline int()/float()s these on every frame, so one bad value would
    make every camera fail and reconnect in a loop.
    """
    bad: list[str] = []
    for keys, conv in ((_INT_KEYS, lambda v: int(float(v))), (_FLOAT_KEYS, float)):
        for k in keys:
            if k in data and data[k] is not None:
                try:
                    data[k] = conv(data[k])
                except (TypeError, ValueError):
                    bad.append(k)
    for k in _BOOL_KEYS:
        if k in data:
            data[k] = _to_bool(data[k])
    return data, bad


def _int_arg(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        return min(hi, max(lo, int(value)))
    except (TypeError, ValueError):
        return default


# ---- request validation ---------------------------------------------------

NAME_MAX = 60
URL_MAX = 500
ZONE_POINTS_MAX = 64
PERSON_CATEGORIES = ("child", "adult", "elderly")
SECRET_SETTINGS = ("twilio_auth_token",)
_TWILIO_SID_RE = re.compile(r"^AC[0-9a-fA-F]{32}$")
_TWILIO_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def _twilio_fields(data: dict[str, Any]) -> None:
    """Check and normalise the Twilio fields of a settings save, in place.

    An empty auth token means "keep the saved one"; twilio_auth_token_clear
    removes it. An empty SID or sender clears that value.
    """
    if "twilio_account_sid" in data:
        sid = str(data["twilio_account_sid"] or "").strip()
        if sid and not _TWILIO_SID_RE.match(sid):
            raise BadInput("The Account SID is AC followed by 32 letters and "
                           "numbers. Copy it from the Twilio Console home page.")
        data["twilio_account_sid"] = sid
    if "twilio_whatsapp_from" in data:
        sender = re.sub(r"[\s\-().]", "", str(data["twilio_whatsapp_from"] or ""))
        sender = sender.removeprefix("whatsapp:")
        if sender and not _E164_RE.match(sender):
            raise BadInput("The WhatsApp sender is a phone number with its "
                           "country code, e.g. +14155238886 for the Twilio sandbox.")
        data["twilio_whatsapp_from"] = sender
    if data.pop("twilio_auth_token_clear", False):
        data["twilio_auth_token"] = ""
    elif "twilio_auth_token" in data:
        token = str(data["twilio_auth_token"] or "").strip()
        if not token:
            data.pop("twilio_auth_token")          # blank field: keep the saved token
        elif not _TWILIO_TOKEN_RE.match(token):
            raise BadInput("The auth token is 32 letters and numbers. Copy it "
                           "from the Twilio Console (Account info).")
        else:
            data["twilio_auth_token"] = token


class BadInput(ValueError):
    """A request field failed validation; answered as HTTP 400."""


def _text(data: dict[str, Any], key: str, label: str, *,
          default: str = "", max_len: int = NAME_MAX) -> str:
    value = str(data.get(key) or "").strip() or default
    if len(value) > max_len:
        raise BadInput(f"{label} is too long ({max_len} characters max)")
    return value


def _category(data: dict[str, Any]) -> str:
    value = str(data.get("category") or "adult").strip().lower()
    if value not in PERSON_CATEGORIES:
        raise BadInput("Category must be child, adult or elderly")
    return value


def _polygon(value: Any) -> list[list[float]]:
    """Validate a zone drawn on the 640x480 reference canvas; clamp to it."""
    if not isinstance(value, list) or not 3 <= len(value) <= ZONE_POINTS_MAX:
        raise BadInput(f"A zone needs 3 to {ZONE_POINTS_MAX} points")
    out = []
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise BadInput("Zone points must be [x, y] pairs")
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            raise BadInput("Zone points must be numbers") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise BadInput("Zone points must be numbers")
        out.append([min(max(x, 0.0), float(ZoneStore.REF_W)),
                    min(max(y, 0.0), float(ZoneStore.REF_H))])
    return out


# ---- factory --------------------------------------------------------------

def create_app(*, db_path: str = "homeshield.db",
               snapshot_dir: str = "snapshots",
               person_photos_dir: str = "person_photos",
               intruder_photos_dir: str = "intruder_photos",
               auto_start: bool = True) -> Flask:

    init_db(db_path)

    snap_path = Path(snapshot_dir).resolve()
    person_dir = Path(person_photos_dir).resolve()
    intruder_dir = Path(intruder_photos_dir).resolve()
    for d in (snap_path, person_dir, intruder_dir):
        d.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["JSON_SORT_KEYS"] = False

    # ---- session / auth config ----
    # Sessions are signed with this key. Keep it in a file beside the
    # database so a restart doesn't sign everyone out ("Remember me").
    secret = os.environ.get("HOMESHIELD_SECRET")
    if not secret:
        key_file = Path(db_path).resolve().with_name("homeshield_secret.key")
        try:
            secret = key_file.read_text(encoding="utf-8").strip()
        except OSError:
            secret = ""
        if not secret:
            secret = secrets.token_hex(32)
            try:
                key_file.write_text(secret, encoding="utf-8")
            except OSError as e:
                log.warning("could not save the session key (%s); sign-ins "
                            "will not survive a restart", e)
    app.secret_key = secret
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # Set HOMESHIELD_COOKIE_SECURE=1 when behind HTTPS.
        SESSION_COOKIE_SECURE=os.environ.get("HOMESHIELD_COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(days=REMEMBER_DAYS),
    )

    user_store = UserStore(db_path)
    app.extensions["homeshield_users"] = user_store
    throttle = LoginThrottle()
    reset_limiter = LoginThrottle()   # reuses the counting: 5 requests / 10 min / address

    @app.errorhandler(BadInput)
    def _bad_input(e: BadInput):
        return jsonify({"error": str(e)}), 400

    # Gzip text for browsers that accept it: the page and the JSON lists
    # shrink 4-5x, which a phone on a hotspot notices. Live streams (MJPEG,
    # server-sent events) and files sent from disk pass through untouched.
    @app.after_request
    def _gzip(resp: Response) -> Response:
        if (resp.status_code != 200 or resp.direct_passthrough or resp.is_streamed
                or resp.mimetype not in _GZIP_TYPES
                or "Content-Encoding" in resp.headers):
            return resp
        resp.vary.add("Accept-Encoding")
        body = resp.get_data()
        if len(body) < _GZIP_MIN_BYTES or not request.accept_encodings["gzip"]:
            return resp
        resp.set_data(gzip.compress(body, compresslevel=6))
        resp.headers["Content-Encoding"] = "gzip"
        return resp

    settings = SettingsStore(db_path)
    bus = EventBus(db_path=db_path, snapshot_dir=snap_path)
    person_store = PersonStore(db_path=db_path, photos_dir=person_dir)
    intruder_store = IntruderStore(db_path=db_path, photos_dir=intruder_dir)
    zone_store = ZoneStore(db_path=db_path)
    incident_store = IncidentStore(db_path)
    try:
        incident_store.backfill()      # group events logged before incidents existed
    except Exception as e:
        log.exception("incident backfill failed: %s", e)
    notifier = WhatsAppNotifier(settings)
    bus.add_incident_listener(notifier.on_incident)
    models = Models(settings=settings)
    manager = CameraManager(
        db_path=db_path, models=models, settings=settings, bus=bus,
        person_store=person_store, intruder_store=intruder_store,
        zone_store=zone_store,
    )

    if auto_start:
        try:
            manager.start()
        except Exception as e:
            log.exception("auto-start failed: %s", e)

    # ===== Pages =========================================================

    @app.route("/")
    def index():
        # The page itself is public: it always renders the shell, then the
        # client-side JS calls /api/me and either shows the login overlay
        # or the dashboard depending on the session. "no-cache" plus an ETag:
        # the browser asks each time, but unless the page changed it gets a
        # 304 with no body instead of the whole page again.
        resp = make_response(render_template("index.html"))
        resp.headers["Cache-Control"] = "no-cache"
        resp.add_etag(weak=True)
        return resp.make_conditional(request)

    # ===== Authentication ===============================================

    def _user_public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "role": row["role"],
            "must_change_password": bool(row.get("must_change", 0)),
        }

    # Public: the sign-in card shows the default login only while it works,
    # and the board's lamp says whether cameras are being watched. Nothing
    # about incidents or cameras: a guest on the Wi-Fi can read this.
    @app.route("/api/setup_state")
    def api_setup_state():
        return jsonify({"default_admin": user_store.default_admin_active(),
                        "monitoring": manager.monitoring()})

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(silent=True) or {}
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        ip = request.remote_addr or ""
        wait = throttle.retry_after(username, ip)
        if wait:
            return jsonify({"error": "Too many wrong passwords.",
                            "retry_after": wait}), 429
        row = user_store.verify(username, password)
        if row is None:
            # Same response for unknown user and wrong password.
            wait = throttle.failed(username, ip)
            if wait:
                return jsonify({"error": "Too many wrong passwords.",
                                "retry_after": wait}), 429
            # Same for real and unknown usernames, so it reveals nothing.
            return jsonify({"error": "invalid credentials",
                            "attempts_left": throttle.remaining(username, ip)}), 401
        throttle.succeeded(username, ip)
        previous = user_store.record_login(row["user_id"], ip)
        login_session(row, remember=_to_bool(data.get("remember", False)))
        return jsonify({"ok": True, **_user_public(row),
                        "last_login": previous or None})

    # Public, like the sign-in form: anyone may ask, and the answer is the
    # same whether or not the username exists, so it can't be used to probe
    # for accounts. Admins see the request in Settings > Users.
    @app.route("/api/password_reset_request", methods=["POST"])
    def api_password_reset_request():
        data = request.get_json(silent=True) or {}
        username = str(data.get("username", "")).strip()[:64]
        ip = request.remote_addr or ""
        if reset_limiter.retry_after("", ip):
            return jsonify({"error": "Too many requests. Try again in a few minutes."}), 429
        reset_limiter.failed("", ip)          # counts every request from this address
        if username:
            user_store.request_reset(username)
        return jsonify({"ok": True})

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        logout_session()
        return jsonify({"ok": True})

    @app.route("/api/me")
    def api_me():
        uid = session_user_id()
        if uid is None:
            return jsonify({"error": "auth_required"}), 401
        row = user_store.get(uid)
        if row is None:
            # User was deleted while logged in.
            logout_session()
            return jsonify({"error": "auth_required"}), 401
        return jsonify({"ok": True, **_user_public(row)})

    @app.route("/api/change_password", methods=["POST"])
    def api_change_password():
        uid = session_user_id()
        if uid is None:
            return jsonify({"error": "auth_required"}), 401
        row = user_store.get(uid)
        if row is None:
            logout_session()
            return jsonify({"error": "auth_required"}), 401
        data = request.get_json(silent=True) or {}
        new_pw = str(data.get("new_password", ""))
        # Require the current password unless the account is flagged for a
        # forced change. Read the flag from the DB, not the session: an admin
        # reset made after this user logged in only exists in the DB, and the
        # change-password screen it triggers doesn't ask for the old password.
        if not row["must_change"]:
            current = str(data.get("current_password", ""))
            if user_store.verify(row["username"], current) is None:
                return jsonify({"error": "Your current password is wrong."}), 400
        # A "new" password equal to the old one (e.g. keeping admin/admin)
        # would leave the account exactly as guessable as before.
        if new_pw.strip().lower() == row["username"].strip().lower():
            return jsonify({"error": "Choose a password that isn't your username."}), 400
        if user_store.verify(row["username"], new_pw) is not None:
            return jsonify({"error": "That's the password you have now. Choose a new one."}), 400
        try:
            user_store.update_password(uid, new_pw)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        # Refresh session to clear must_change; keep the "Remember me" choice.
        row = user_store.get(uid)
        if row is not None:
            login_session(row, remember=bool(session.permanent))
        return jsonify({"ok": True})

    # ===== User management (admin only) =================================

    @app.route("/api/users")
    @require_admin
    def api_users_list():
        return jsonify({"users": user_store.list_users()})

    @app.route("/api/users", methods=["POST"])
    @require_admin
    def api_users_add():
        data = request.get_json(silent=True) or {}
        try:
            info = user_store.create_user(
                username=str(data.get("username", "")),
                password=str(data.get("password", "")),
                role=str(data.get("role", ROLE_GUEST)),
                must_change=bool(data.get("must_change_password", False)),
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, **info})

    @app.route("/api/users/<int:uid>", methods=["DELETE"])
    @require_admin
    def api_users_delete(uid: int):
        if uid == session_user_id():
            return jsonify({"error": "You can't delete the account you're signed in with."}), 400
        try:
            ok = user_store.delete_user(uid)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not ok:
            return jsonify({"error": "That user no longer exists."}), 404
        return jsonify({"ok": True})

    @app.route("/api/users/<int:uid>/role", methods=["POST"])
    @require_admin
    def api_users_set_role(uid: int):
        data = request.get_json(silent=True) or {}
        role = str(data.get("role", "")).strip().lower()
        if uid == session_user_id() and role != ROLE_ADMIN:
            return jsonify({"error": "You can't remove your own admin role."}), 400
        try:
            ok = user_store.update_role(uid, role)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not ok:
            return jsonify({"error": "That user no longer exists."}), 404
        return jsonify({"ok": True})

    @app.route("/api/users/<int:uid>/password", methods=["POST"])
    @require_admin
    def api_users_set_password(uid: int):
        data = request.get_json(silent=True) or {}
        new_pw = str(data.get("password", ""))
        try:
            ok = user_store.update_password(uid, new_pw)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not ok:
            return jsonify({"error": "That user no longer exists."}), 404
        # Admin-issued resets force the user to pick their own next time.
        with write_conn(db_path) as conn:
            conn.execute(
                "UPDATE users SET must_change = 1 WHERE user_id = ?", (int(uid),)
            )
        return jsonify({"ok": True})

    # ===== Status & system ==============================================

    def _is_admin() -> bool:
        me = user_store.get(session_user_id()) or {}
        return me.get("role") == ROLE_ADMIN

    @app.route("/api/status")
    @require_login
    def api_status():
        out = manager.status()
        if not _is_admin():   # camera sources can carry RTSP passwords
            for cam in out.get("cameras", {}).values():
                cam.pop("url", None)
        s = settings.all()
        out['fall_enabled'] = bool(s.get('fall_enabled', True))
        out['fire_enabled'] = bool(s.get('fire_enabled', True))
        out['face_enabled'] = bool(s.get('face_enabled', True))
        out.update(incident_store.open_counts())
        return jsonify(out)

    @app.route("/api/system/start", methods=["POST"])
    @require_admin
    def api_start():
        manager.start()
        return jsonify({"ok": True, "running": manager.is_running()})

    @app.route("/api/system/stop", methods=["POST"])
    @require_admin
    def api_stop():
        manager.stop()
        return jsonify({"ok": True, "running": manager.is_running()})

    @app.route("/healthz")
    @require_login
    def healthz():
        return jsonify({
            "ok": True,
            "running": manager.is_running(),
            "models": {
                "pose_loaded": models.pose_model is not None,
                "fire_loaded": models.fire_model is not None,
                "face_available": bool(models.face_engine
                                       and models.face_engine.available),
                "face_device": (models.face_engine.device
                                if models.face_engine else "none"),
                "device": models.device,
            },
        })

    # ===== Cameras ======================================================

    @app.route("/api/cameras")
    @require_login
    def api_cameras_list():
        cams = manager.store.list()
        if not _is_admin():   # camera sources can carry RTSP passwords
            cams = [{k: v for k, v in c.items() if k != "url"} for c in cams]
        return jsonify(cams)

    @app.route("/api/cameras", methods=["POST"])
    @require_admin
    def api_cameras_add():
        data = request.get_json(silent=True) or {}
        name = _text(data, "name", "Camera name", default="Camera")
        url = _strip_quotes(str(data.get("url") or ""))
        if not url:
            raise BadInput("Enter a camera source: 0 for the built-in webcam, "
                           "or an rtsp:// or http:// address")
        if len(url) > URL_MAX:
            raise BadInput(f"Camera source is too long ({URL_MAX} characters max)")
        location = _text(data, "location", "Location")
        cid = manager.add_camera(name=name, url=url, location=location)
        return jsonify({"camera_id": cid, "ok": True})

    @app.route("/api/cameras/<int:cid>", methods=["DELETE"])
    @require_admin
    def api_cameras_delete(cid: int):
        manager.delete_camera(cid)
        return jsonify({"ok": True})

    @app.route("/api/models")
    @require_admin
    def api_models():
        return jsonify({
            "pose": list_pose_models(),
            "fire": list_fire_models(),
        })

    # ===== Video / snapshots ============================================

    @app.route("/video_feed/<int:cid>")
    @require_login
    def video_feed(cid: int):
        latest = manager.latest(cid)
        if latest is None:
            return abort(404)

        def gen():
            version = -1
            while True:
                jpeg, version = latest.get_blocking(version, timeout=1.0)
                # Camera removed or system stopped: end the stream instead of
                # replaying its last frame to the open tab forever.
                if manager.latest(cid) is not latest:
                    return
                if jpeg is None:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # Login, not admin: guests already watch /video_feed, and the dashboard
    # shows stills instead of extra streams to stay under the browser's
    # per-server connection limit.
    @app.route("/frame_snap/<int:cid>")
    @require_login
    def frame_snap(cid: int):
        latest = manager.latest(cid)
        if latest is None:
            return abort(404)
        jpeg = latest.jpeg()
        if jpeg is None:
            return abort(404)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/snapshots/<path:fname>")
    @require_login
    def snapshot_file(fname: str):
        return send_from_directory(snap_path, fname)

    @app.route("/person_photos/<path:fname>")
    @require_admin
    def person_photo(fname: str):
        return send_from_directory(person_dir, fname)

    @app.route("/intruder_photos/<path:fname>")
    @require_admin
    def intruder_photo(fname: str):
        return send_from_directory(intruder_dir, fname)

    # ===== Events =======================================================

    @app.route("/api/events")
    @require_login
    def api_events():
        limit = _int_arg(request.args.get("limit"), 50, 1, 1000)
        etype = request.args.get("type") or None
        return jsonify(bus.list(limit=limit, event_type=etype))

    @app.route("/api/events/clear", methods=["POST"])
    @require_admin
    def api_events_clear():
        return jsonify({"ok": True, "deleted": bus.clear()})

    @app.route("/events_stream")
    @require_login
    def events_stream():
        def gen():
            q = bus.subscribe()
            try:
                yield "retry: 3000\n\n"
                yield f"event: hello\ndata: {json.dumps({'ok': True})}\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15.0)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    if isinstance(ev, Event):
                        yield f"event: alert\ndata: {json.dumps(ev.to_json())}\n\n"
                    else:
                        yield f"event: {ev['sse']}\ndata: {json.dumps(ev['data'])}\n\n"
            finally:
                bus.unsubscribe(q)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # ===== Incidents ====================================================
    # Anyone signed in may close an incident (family members handle alerts
    # at home); the record keeps who did it. Clearing stays admin-only.

    @app.route("/api/incidents/board")
    @require_login
    def api_incidents_board():
        recent = _int_arg(request.args.get("recent"), 6, 0, 50)
        return jsonify(incident_store.board(recent=recent))

    @app.route("/api/incidents")
    @require_login
    def api_incidents():
        status = request.args.get("status") or None
        if status not in (None, "resolved", *STATUSES):
            raise BadInput("Unknown status filter")
        kind = request.args.get("kind") or None
        if kind not in (None, *KINDS):
            raise BadInput("Unknown incident type")
        cam = request.args.get("camera_id")
        camera_id = _int_arg(cam, -1, 0, 2**31 - 1) if cam else None
        limit = _int_arg(request.args.get("limit"), 50, 1, 500)
        return jsonify(incident_store.list(status=status, kind=kind,
                                           camera_id=camera_id, limit=limit))

    @app.route("/api/incidents/stats")
    @require_login
    def api_incidents_stats():
        days = _int_arg(request.args.get("days"), 7, 1, 365)
        return jsonify(incident_store.stats(days=days))

    @app.route("/api/incidents/<int:iid>")
    @require_login
    def api_incident(iid: int):
        inc = incident_store.get(iid, with_events=True)
        if inc is None:
            return jsonify({"error": "That incident no longer exists"}), 404
        return jsonify(inc)

    @app.route("/api/incidents/<int:iid>/resolve", methods=["POST"])
    @require_login
    def api_incident_resolve(iid: int):
        data = request.get_json(silent=True) or {}
        outcome = str(data.get("outcome") or "")
        if outcome not in OUTCOMES:
            raise BadInput("Choose Acknowledge or False alarm")
        note = _text(data, "note", "Note", max_len=NOTE_MAX)
        me = user_store.get(session_user_id()) or {}
        inc = incident_store.resolve(iid, outcome=outcome, note=note,
                                     by=me.get("username") or "unknown")
        if inc is None:
            return jsonify({"error": "That incident no longer exists"}), 404
        bus.broadcast("incident", {**inc, "change": "resolved"})
        return jsonify(inc)

    # ===== Notifications ================================================

    @app.route("/api/notify/status")
    @require_admin
    def api_notify_status():
        return jsonify(notifier.status())

    @app.route("/api/notify/test", methods=["POST"])
    @require_admin
    def api_notify_test():
        st = notifier.status()
        if st["missing"]:
            raise BadInput("Twilio is not set up yet. Fill in " + ", ".join(st["missing"])
                           + " under Twilio account and save, then try again.")
        if not st["phones"]:
            raise BadInput("Add at least one phone number with its country "
                           "code (e.g. +60123456789) and save first")
        return jsonify(notifier.send_test())

    # ===== Zones ========================================================

    @app.route("/api/zones")
    @require_admin
    def api_zones_list():
        return jsonify(zone_store.list_all())

    @app.route("/api/zones", methods=["POST"])
    @require_admin
    def api_zones_add():
        data = request.get_json(silent=True) or {}
        cam_id = _int_arg(data.get("camera_id"), -1, 0, 2**31 - 1)
        if cam_id < 0:
            return jsonify({"error": "Pick a camera first."}), 400
        if cam_id not in {c["camera_id"] for c in manager.store.list()}:
            return jsonify({"error": "That camera no longer exists"}), 400
        zid = zone_store.add(
            zone_name=_text(data, "zone_name", "Zone name", default="Zone"),
            camera_id=cam_id,
            polygon=_polygon(data.get("polygon")),
            zone_type=str(data.get("zone_type", "danger")),
        )
        return jsonify({"ok": True, "zone_id": zid})

    @app.route("/api/zones/<int:zid>", methods=["DELETE"])
    @require_admin
    def api_zones_delete(zid: int):
        zone_store.delete(zid)
        return jsonify({"ok": True})

    # ===== Persons ======================================================

    def _face_ok(face: dict[str, Any]) -> bool:
        return is_good_face(
            face,
            min_size=float(settings.get("face_min_size", 40)),
            min_det_score=float(settings.get("face_min_det_score", 0.6)),
        )

    @app.route("/api/persons")
    @require_admin
    def api_persons():
        return jsonify({
            "face_rec_enabled": bool(models.face_engine
                                     and models.face_engine.available),
            # Lets the page tell "switched off in Settings" from "failed to load".
            "face_setting": bool(settings.get("face_enabled", True)),
            "persons": person_store.list(),
        })

    @app.route("/api/persons", methods=["POST"])
    @require_admin
    def api_persons_add():
        data = request.get_json(silent=True) or {}
        name = _text(data, "name", "Name")
        category = _category(data)
        cid = data.get("camera_id")
        if not name:
            return jsonify({"error": "Enter the person's name."}), 400
        if not models.face_engine or not models.face_engine.available:
            return jsonify({"error": "Face recognition is off, so nobody can be "
                                     "registered. Turn it on in Settings > Face "
                                     "recognition. "
                                     "Install insightface + onnxruntime."}), 400
        if cid is None:
            return jsonify({"error": "Pick a camera to capture from."}), 400
        cam_id = _int_arg(cid, -1, 0, 2**31 - 1)
        latest = manager.latest(cam_id) if cam_id >= 0 else None
        if latest is None:
            return jsonify({"error": "That camera isn't running. Start monitoring "
                                     "or check the camera, then try again."}), 400
        frame = latest.raw()
        if frame is None:
            return jsonify({"error": "No picture from that camera yet. "
                                     "Wait a moment and try again."}), 400
        face = models.face_engine.best_face(frame)
        if not face or face.get("embedding") is None:
            return jsonify({"error": "No face in view. Ask the person to look "
                                     "straight at the camera."}), 400
        if not _face_ok(face):
            return jsonify({"error": "The face is too small or blurry. Move closer "
                                     "and face the camera directly."}), 400
        info = person_store.add(
            name=name, category=category,
            embedding=face["embedding"],
            frame_bgr=frame,
            face_bbox=(face["x"], face["y"], face["w"], face["h"]),
        )
        info["detected_age"] = face.get("age")
        return jsonify(info)

    @app.route("/api/persons/<int:pid>", methods=["DELETE"])
    @require_admin
    def api_persons_delete(pid: int):
        person_store.delete(pid)
        return jsonify({"ok": True})

    @app.route("/api/detect_face/<int:cid>")
    @require_admin
    def api_detect_face(cid: int):
        latest = manager.latest(int(cid))
        if latest is None:
            return jsonify({"error": "no_frame"})
        frame = latest.raw()
        if frame is None:
            return jsonify({"error": "no_frame"})
        if not models.face_engine or not models.face_engine.available:
            return jsonify({"error": "face_rec_disabled",
                            "width": frame.shape[1],
                            "height": frame.shape[0]})
        face = models.face_engine.best_face(frame)
        h, w = frame.shape[:2]
        if not face:
            return jsonify({"face": None, "width": w, "height": h})
        return jsonify({
            "face": {
                "x": face["x"], "y": face["y"],
                "w": face["w"], "h": face["h"],
                "age": face.get("age"),
                # Same gate as enrolment, so the preview never offers to
                # capture a face the server would then reject.
                "quality_ok": _face_ok(face),
            },
            "width": w, "height": h,
        })

    # ===== Intruders ====================================================

    @app.route("/api/intruders")
    @require_admin
    def api_intruders():
        include = request.args.get("include_dismissed") in ("1", "true", "yes")
        return jsonify(intruder_store.list(include_dismissed=include))

    @app.route("/api/intruders/<int:iid>/dismiss", methods=["POST"])
    @require_admin
    def api_intruders_dismiss(iid: int):
        intruder_store.dismiss(iid)
        return jsonify({"ok": True, "intruder_id": iid})

    @app.route("/api/intruders/<int:iid>/register", methods=["POST"])
    @require_admin
    def api_intruders_register(iid: int):
        data = request.get_json(silent=True) or {}
        name = _text(data, "name", "Name")
        category = _category(data)
        if not name:
            return jsonify({"error": "Enter the person's name."}), 400
        rec = intruder_store.get(iid)
        if rec is None:
            return jsonify({"error": "That intruder record no longer exists."}), 404
        if rec.get("embedding") is None:
            return jsonify({"error": "This record has no face data to "
                                     "register from."}), 400
        info = person_store.add(
            name=name, category=category,
            embedding=rec["embedding"],
            frame_bgr=None, face_bbox=None,
        )
        intruder_store.delete(iid)
        return jsonify({"ok": True, **info})

    @app.route("/api/intruders/<int:iid>", methods=["DELETE"])
    @require_admin
    def api_intruders_delete(iid: int):
        intruder_store.delete(iid)
        return jsonify({"ok": True})

    # ===== Settings =====================================================

    def _public_settings() -> dict[str, Any]:
        """All settings except secrets, plus whether a token is saved."""
        out = settings.all()
        token = str(out.get("twilio_auth_token") or "")
        for k in SECRET_SETTINGS:
            out.pop(k, None)
        out["twilio_token_saved"] = bool(token)
        out["twilio_token_hint"] = token[-4:] if len(token) >= 8 else ""
        return out

    @app.route("/api/settings")
    @require_admin
    def api_settings_get():
        return jsonify(_public_settings())

    @app.route("/api/settings", methods=["POST"])
    @require_admin
    def api_settings_post():
        data, bad = _coerce_settings(request.get_json(silent=True) or {})
        if bad:
            return jsonify({"error": "These settings need a number: "
                                     + ", ".join(bad)}), 400
        for k in ("twilio_token_saved", "twilio_token_hint"):   # read-only fields
            data.pop(k, None)
        _twilio_fields(data)
        settings.update(data)
        manager.reload_settings()
        return jsonify(_public_settings())

    return app
