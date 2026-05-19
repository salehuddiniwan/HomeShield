"""Flask app implementing the HomeShield dashboard API."""

from __future__ import annotations

import json
import os
import queue
import secrets
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_from_directory, session)

from .auth import (ROLE_ADMIN, ROLE_GUEST, UserStore, login_session,
                   logout_session, require_admin, require_login,
                   session_must_change, session_role, session_user_id)
from .cameras import CameraManager
from .db import init_db, write_conn
from .events import Event, EventBus
from .persons import IntruderStore, PersonStore
from .pipeline import Models, list_fire_models, list_pose_models
from .settings import SettingsStore
from .zones import ZoneStore


# ---- settings POST coercion ----------------------------------------------

_INT_KEYS = ("inactivity_seconds", "alert_cooldown", "yolo_imgsz",
             "process_fps", "fire_cooldown", "intruder_cooldown",
             "fire_every_n", "face_every_n")
_FLOAT_KEYS = ("fall_threshold", "yolo_confidence", "fire_confidence",
               "face_match_threshold")
_BOOL_KEYS = ("use_fp16", "fall_enabled", "fire_enabled", "face_enabled")


def _coerce_settings(data: dict[str, Any]) -> dict[str, Any]:
    for k in _INT_KEYS:
        if k in data and data[k] is not None:
            try:
                data[k] = int(data[k])
            except Exception:
                pass
    for k in _FLOAT_KEYS:
        if k in data and data[k] is not None:
            try:
                data[k] = float(data[k])
            except Exception:
                pass
    for k in _BOOL_KEYS:
        if k in data:
            data[k] = bool(data[k])
    return data


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
    secret = os.environ.get("HOMESHIELD_SECRET")
    if not secret:
        secret = secrets.token_hex(32)
        print("[server] HOMESHIELD_SECRET not set; using an ephemeral key "
              "(sessions will not survive a restart)")
    app.secret_key = secret
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # Set HOMESHIELD_COOKIE_SECURE=1 when behind HTTPS.
        SESSION_COOKIE_SECURE=os.environ.get("HOMESHIELD_COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    )

    user_store = UserStore(db_path)

    settings = SettingsStore(db_path)
    bus = EventBus(db_path=db_path, snapshot_dir=snap_path)
    person_store = PersonStore(db_path=db_path, photos_dir=person_dir)
    intruder_store = IntruderStore(db_path=db_path, photos_dir=intruder_dir)
    zone_store = ZoneStore(db_path=db_path)
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
            print(f"[server] auto-start failed: {e}")

    # ===== Pages =========================================================

    @app.route("/")
    def index():
        # The page itself is public: it always renders the shell, then the
        # client-side JS calls /api/me and either shows the login overlay
        # or the dashboard depending on the session.
        return render_template("index.html")

    # ===== Authentication ===============================================

    def _user_public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "role": row["role"],
            "must_change_password": bool(row.get("must_change", 0)),
        }

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(silent=True) or {}
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        row = user_store.verify(username, password)
        if row is None:
            # Same response for unknown user and wrong password.
            return jsonify({"error": "invalid credentials"}), 401
        login_session(row)
        return jsonify({"ok": True, **_user_public(row)})

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
        data = request.get_json(silent=True) or {}
        new_pw = str(data.get("new_password", ""))
        # If the user is NOT on a must-change flow, require their current pw.
        if not session_must_change():
            current = str(data.get("current_password", ""))
            row = user_store.get(uid)
            if row is None:
                return jsonify({"error": "auth_required"}), 401
            verified = user_store.verify(row["username"], current)
            if verified is None:
                return jsonify({"error": "current password is incorrect"}), 400
        try:
            user_store.update_password(uid, new_pw)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        # Refresh session to clear must_change.
        row = user_store.get(uid)
        if row is not None:
            login_session(row)
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
            return jsonify({"error": "cannot delete your own account"}), 400
        try:
            ok = user_store.delete_user(uid)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not ok:
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True})

    @app.route("/api/users/<int:uid>/role", methods=["POST"])
    @require_admin
    def api_users_set_role(uid: int):
        data = request.get_json(silent=True) or {}
        role = str(data.get("role", "")).strip().lower()
        if uid == session_user_id() and role != ROLE_ADMIN:
            return jsonify({"error": "cannot demote yourself"}), 400
        try:
            ok = user_store.update_role(uid, role)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not ok:
            return jsonify({"error": "not found"}), 404
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
            return jsonify({"error": "not found"}), 404
        # Admin-issued resets force the user to pick their own next time.
        with write_conn(db_path) as conn:
            conn.execute(
                "UPDATE users SET must_change = 1 WHERE user_id = ?", (int(uid),)
            )
        return jsonify({"ok": True})

    # ===== Status & system ==============================================

    @app.route("/api/status")
    @require_login
    def api_status():
        return jsonify(manager.status())

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
                "device": models.device,
            },
        })

    # ===== Cameras ======================================================

    @app.route("/api/cameras")
    @require_login
    def api_cameras_list():
        return jsonify(manager.store.list())

    @app.route("/api/cameras", methods=["POST"])
    @require_admin
    def api_cameras_add():
        data = request.get_json(silent=True) or {}
        cid = manager.add_camera(
            name=str(data.get("name", "Camera")),
            url=str(data.get("url", "0")),
            location=str(data.get("location", "")),
        )
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
                if jpeg is None:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/frame_snap/<int:cid>")
    @require_admin
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

    # Icons are part of the page chrome (login + dashboard); leave them public
    # so the login screen can render its logo before the user authenticates.
    icon_dir = (Path(__file__).resolve().parent.parent / "Icon").resolve()
    if icon_dir.is_dir():
        @app.route("/icons/<path:fname>")
        def icon_file(fname: str):
            return send_from_directory(icon_dir, fname)

    # ===== Events =======================================================

    @app.route("/api/events")
    @require_login
    def api_events():
        limit = int(request.args.get("limit", 50))
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
                    yield f"event: alert\ndata: {json.dumps(ev.to_json())}\n\n"
            finally:
                bus.unsubscribe(q)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    # ===== Zones ========================================================

    @app.route("/api/zones")
    @require_admin
    def api_zones_list():
        return jsonify(zone_store.list_all())

    @app.route("/api/zones", methods=["POST"])
    @require_admin
    def api_zones_add():
        data = request.get_json(silent=True) or {}
        zid = zone_store.add(
            zone_name=str(data.get("zone_name", "Zone")),
            camera_id=int(data.get("camera_id", 0)),
            polygon=list(data.get("polygon") or []),
            zone_type=str(data.get("zone_type", "danger")),
        )
        return jsonify({"ok": True, "zone_id": zid})

    @app.route("/api/zones/<int:zid>", methods=["DELETE"])
    @require_admin
    def api_zones_delete(zid: int):
        zone_store.delete(zid)
        return jsonify({"ok": True})

    # ===== Persons ======================================================

    @app.route("/api/persons")
    @require_admin
    def api_persons():
        return jsonify({
            "face_rec_enabled": bool(models.face_engine
                                     and models.face_engine.available),
            "persons": person_store.list(),
        })

    @app.route("/api/persons", methods=["POST"])
    @require_admin
    def api_persons_add():
        data = request.get_json(silent=True) or {}
        name = str(data.get("name", "")).strip()
        category = str(data.get("category", "adult"))
        cid = data.get("camera_id")
        if not name:
            return jsonify({"error": "Name required"}), 400
        if not models.face_engine or not models.face_engine.available:
            return jsonify({"error": "Face recognition not available. "
                                     "Install insightface + onnxruntime."}), 400
        if cid is None:
            return jsonify({"error": "Select a camera"}), 400
        latest = manager.latest(int(cid))
        if latest is None:
            return jsonify({"error": "Camera not running"}), 400
        frame = latest.raw()
        if frame is None:
            return jsonify({"error": "No frame from camera"}), 400
        face = models.face_engine.best_face(frame)
        if not face or face.get("embedding") is None:
            return jsonify({"error": "No face found in frame"}), 400
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
        name = str(data.get("name", "")).strip()
        category = str(data.get("category", "adult"))
        if not name:
            return jsonify({"error": "Name required"}), 400
        rec = intruder_store.get(iid)
        if rec is None:
            return jsonify({"error": "Intruder not found"}), 404
        if rec.get("embedding") is None:
            return jsonify({"error": "Intruder has no embedding"}), 400
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

    @app.route("/api/settings")
    @require_admin
    def api_settings_get():
        return jsonify(settings.all())

    @app.route("/api/settings", methods=["POST"])
    @require_admin
    def api_settings_post():
        data = _coerce_settings(request.get_json(silent=True) or {})
        out = settings.update(data)
        manager.reload_settings()
        return jsonify(out)

    return app
