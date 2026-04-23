"""
HomeShield — Main Flask Application
Intelligent ML-Powered CCTV Surveillance for Elderly & Child Safety
"""
import os
import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import (Flask, render_template, Response, jsonify, request,
                   send_from_directory, redirect, url_for)
from config import Config
from database import Database
from detector import Detector
from camera_manager import CameraManager
from alerter import Alerter

# ── App init ─────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

db = Database()

# Apply any saved settings from DB into Config BEFORE detector loads
# so the correct model is used on startup
def _apply_saved_settings():
    _map = {
        "yolo_model":       ("YOLO_MODEL",                str),
        "yolo_confidence":  ("YOLO_CONFIDENCE",           float),
        "yolo_imgsz":       ("YOLO_IMGSZ",                int),
        "process_fps":      ("PROCESS_FPS",               int),
        "use_fp16":         ("USE_FP16",                  lambda v: str(v).lower() == "true"),
        "fall_threshold":   ("FALL_CONFIDENCE_THRESHOLD", float),
        "inactivity_seconds": ("INACTIVITY_SECONDS",      int),
        "alert_cooldown":   ("ALERT_COOLDOWN_SECONDS",    int),
    }
    for db_key, (cfg_attr, cast) in _map.items():
        val = db.get_setting(db_key, None)
        if val is not None:
            try:
                setattr(Config, cfg_attr, cast(val))
            except Exception:
                pass

_apply_saved_settings()
print(f"[INFO] Active model: {Config.YOLO_MODEL}")

detector = Detector()
cam_manager = CameraManager()
alerter = Alerter()

os.makedirs(Config.SNAPSHOT_DIR, exist_ok=True)

# Global state
system_state = {
    "running": False,
    "people_count": 0,
    "active_alerts": [],
}


# ── Processing pipeline ─────────────────────────────────────
def processing_loop():
    """
    GPU-optimised main loop.

    All active camera frames are collected first, then sent to the detector
    as a single batch so YOLOv8 executes one GPU forward pass per tick
    instead of one per camera.  Alert handling is unchanged.
    """
    frame_interval = 1.0 / Config.PROCESS_FPS

    while system_state["running"]:
        loop_start = time.time()

        # ── Collect frames from all active cameras ────────────────────────
        active = list(cam_manager.get_all_active().items())
        batch_inputs = []   # (frame, cid, zones)
        batch_cams   = []   # matching CameraStream objects

        for cid, cam in active:
            grabbed, frame = cam.read()
            if not grabbed or frame is None:
                continue
            zones = db.get_zones(camera_id=cid)
            batch_inputs.append((frame, cid, zones))
            batch_cams.append((cid, cam))

        # ── Single batched GPU inference pass ─────────────────────────────
        if batch_inputs:
            batch_results = detector.process_frames_batch(batch_inputs)
        else:
            batch_results = []

        total_people = 0
        for (cid, cam), (annotated, events) in zip(batch_cams, batch_results):
            cam.annotated_frame = annotated
            total_people += len(detector.tracker.objects)

            # Handle events
            for event in events:
                etype = event["event_type"]
                if not alerter.should_alert(etype, cid):
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"{etype}_{cid}_{ts}.jpg"
                snap_path = os.path.join(Config.SNAPSHOT_DIR, snap_name)
                cv2.imwrite(snap_path, annotated)
                # Forward slashes so the URL works on Windows
                snap_url_path = snap_path.replace("\\", "/")

                details = event.get("zone_name", "")
                db.log_event(
                    event_type=etype,
                    camera_id=cid,
                    camera_name=cam.name,
                    person_category=event.get("person_category", "unknown"),
                    confidence=event.get("confidence", 0),
                    snapshot_path=snap_url_path,
                    alert_sent=True,
                    details=details,
                )
                alerter.send_alert(
                    event_type=etype,
                    camera_name=f"{cam.name} ({cam.location})",
                    person_category=event.get("person_category", "unknown"),
                    confidence=event.get("confidence", 0),
                    snapshot_path=snap_path,
                    camera_id=cid,
                    details=details,
                )

        system_state["people_count"] = total_people

        elapsed = time.time() - loop_start
        time.sleep(max(0, frame_interval - elapsed))


def start_system():
    global detector
    if system_state["running"]:
        return

    detector = Detector()

    # Load cameras from DB
    cameras = db.get_cameras(active_only=True)
    if not cameras:
        # Add default camera if none exist
        for cam in Config.DEFAULT_CAMERAS:
            cid = db.add_camera(cam["name"], cam["url"], cam["location"])
            cam_manager.add_camera(cid, cam["name"], cam["url"], cam["location"])
    else:
        for cam in cameras:
            cam_manager.add_camera(cam["camera_id"], cam["name"], cam["url"], cam["location"])

    system_state["running"] = True
    threading.Thread(target=processing_loop, daemon=True).start()
    print("[INFO] HomeShield system started")


def stop_system():
    system_state["running"] = False
    cam_manager.stop_all()
    print("[INFO] HomeShield system stopped")


# ── Video streaming ──────────────────────────────────────────
def generate_feed(camera_id):
    """MJPEG generator for a single camera."""
    while True:
        cam = cam_manager.cameras.get(camera_id)
        if cam is None:
            time.sleep(0.5)
            continue

        frame = getattr(cam, "annotated_frame", None)
        if frame is None:
            grabbed, frame = cam.read()
            if not grabbed or frame is None:
                time.sleep(0.1)
                continue

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(1.0 / 15)  # 15fps stream


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    return Response(
        generate_feed(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/frame_snap/<int:camera_id>")
def frame_snap(camera_id):
    """Return a single JPEG frame for the zone editor canvas."""
    grabbed, frame = cam_manager.get_frame(camera_id)
    if not grabbed or frame is None:
        # Return a small black placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", placeholder)
    else:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/status")
def api_status():
    cameras = cam_manager.get_status()
    online = sum(1 for c in cameras.values() if c["active"])
    return jsonify({
        "running": system_state["running"],
        "cameras_online": online,
        "cameras_total": len(cameras),
        "cameras": cameras,
        "people_count": system_state["people_count"],
        "alerts_today": db.get_today_alert_count(),
        "model": db.get_setting("yolo_model", Config.YOLO_MODEL).replace(".pt", "-pose.pt").replace("-pose-pose", "-pose"),
    })


@app.route("/api/events")
def api_events():
    limit = request.args.get("limit", 50, type=int)
    etype = request.args.get("type", None)
    cid = request.args.get("camera_id", None, type=int)
    events = db.get_events(limit=limit, event_type=etype, camera_id=cid)
    return jsonify(events)


@app.route("/api/cameras", methods=["GET"])
def api_get_cameras():
    return jsonify(db.get_cameras())


@app.route("/api/cameras", methods=["POST"])
def api_add_camera():
    data = request.json
    name = data.get("name", "Camera")
    url = data.get("url", "0")
    location = data.get("location", "")
    cid = db.add_camera(name, url, location)
    if system_state["running"]:
        cam_manager.add_camera(cid, name, url, location)
    return jsonify({"camera_id": cid, "status": "added"})


@app.route("/api/cameras/<int:camera_id>", methods=["DELETE"])
def api_delete_camera(camera_id):
    db.delete_camera(camera_id)
    cam_manager.remove_camera(camera_id)
    return jsonify({"status": "deleted"})


@app.route("/api/zones", methods=["GET"])
def api_get_zones():
    cid = request.args.get("camera_id", None, type=int)
    return jsonify(db.get_zones(camera_id=cid))


@app.route("/api/zones", methods=["POST"])
def api_add_zone():
    data = request.json
    zone_name = data.get("zone_name", "Danger Zone")
    camera_id = data.get("camera_id")
    polygon = data.get("polygon", [])
    if not camera_id or not polygon:
        return jsonify({"error": "camera_id and polygon required"}), 400
    zid = db.add_zone(zone_name, camera_id, polygon)
    return jsonify({"zone_id": zid, "status": "added"})


@app.route("/api/zones/<int:zone_id>", methods=["DELETE"])
def api_delete_zone(zone_id):
    db.delete_zone(zone_id)
    return jsonify({"status": "deleted"})


@app.route("/api/system/start", methods=["POST"])
def api_start():
    start_system()
    return jsonify({"status": "started"})


@app.route("/api/system/stop", methods=["POST"])
def api_stop():
    stop_system()
    return jsonify({"status": "stopped"})


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify({
        "fall_threshold":   float(db.get_setting("fall_threshold",   Config.FALL_CONFIDENCE_THRESHOLD)),
        "inactivity_seconds": int(db.get_setting("inactivity_seconds", Config.INACTIVITY_SECONDS)),
        "alert_cooldown":   int(db.get_setting("alert_cooldown",     Config.ALERT_COOLDOWN_SECONDS)),
        "alert_phones":     db.get_setting("alert_phones",           ",".join(Config.ALERT_PHONE_NUMBERS)),
        # Model & GPU
        "yolo_model":       db.get_setting("yolo_model",      Config.YOLO_MODEL),
        "yolo_confidence":  float(db.get_setting("yolo_confidence", Config.YOLO_CONFIDENCE)),
        "yolo_imgsz":       int(db.get_setting("yolo_imgsz",   Config.YOLO_IMGSZ)),
        "process_fps":      int(db.get_setting("process_fps",  Config.PROCESS_FPS)),
        "use_fp16":         db.get_setting("use_fp16", str(Config.USE_FP16)).lower() == "true",
    })


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    data = request.json
    all_keys = ("fall_threshold", "inactivity_seconds", "alert_cooldown",
                "alert_phones", "yolo_model", "yolo_confidence",
                "yolo_imgsz", "process_fps", "use_fp16")
    for key in all_keys:
        if key in data:
            db.set_setting(key, str(data[key]))

    # Apply instantly to runtime config where possible
    if "fall_threshold" in data:
        Config.FALL_CONFIDENCE_THRESHOLD = float(data["fall_threshold"])
    if "inactivity_seconds" in data:
        Config.INACTIVITY_SECONDS = int(data["inactivity_seconds"])
    if "alert_cooldown" in data:
        Config.ALERT_COOLDOWN_SECONDS = int(data["alert_cooldown"])
    if "alert_phones" in data:
        Config.ALERT_PHONE_NUMBERS = [p.strip() for p in data["alert_phones"].split(",") if p.strip()]
    if "yolo_confidence" in data:
        Config.YOLO_CONFIDENCE = float(data["yolo_confidence"])
    if "process_fps" in data:
        Config.PROCESS_FPS = int(data["process_fps"])
    # yolo_model, yolo_imgsz, use_fp16 require system restart to take effect
    if "yolo_model" in data:
        Config.YOLO_MODEL = data["yolo_model"]
    if "yolo_imgsz" in data:
        Config.YOLO_IMGSZ = int(data["yolo_imgsz"])
    if "use_fp16" in data:
        Config.USE_FP16 = str(data["use_fp16"]).lower() == "true"

    return jsonify({"status": "updated"})


@app.route("/api/events/clear", methods=["POST"])
def api_clear_events():
    db.clear_events()
    system_state["people_count"] = 0
    return jsonify({"status": "cleared"})


@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(Config.SNAPSHOT_DIR, filename)


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════╗
    ║         🛡️  HomeShield v1.0                  ║
    ║  ML-Powered CCTV Surveillance System         ║
    ║  Elderly & Child Safety Monitoring           ║
    ╚══════════════════════════════════════════════╝
    """)

    start_system()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=False,  # debug=False with threaded camera capture
        threaded=True,
    )