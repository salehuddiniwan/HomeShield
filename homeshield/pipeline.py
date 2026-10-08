"""Per-camera detection pipeline. Models are loaded once (in `Models`) and
shared across cameras; per-camera mutable state lives in `CameraPipeline`.

Stages per frame:
  1. Pose      (YOLO pose + ByteTrack + research-tuned FSM, every frame)
  2. Face      (InsightFace on a worker thread, every face_every_n frames;
                recognised faces are attached to the tracked person)
  3. Zone      (point-in-polygon vs configured zones)
  4. Fire      (YOLO detect, every fire_every_n frames, k-of-n confirmed)

Edge-triggered events with cooldowns:
  fall_detected, lying_motionless, inactivity, zone_entry,
  intruder_detected, fire_detected.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import paths
from .detectors import fire
from .detectors.face import FaceEngine, best_match, is_good_face
from .detectors.fall import Config, MultiPersonState, State
from .events import Event
from .zones import point_in_polygon, scale_polygon

log = logging.getLogger(__name__)


# ---- weights resolution ---------------------------------------------------

def resolve_weights(rel_or_abs: str, search_roots: list[Path]) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute() and p.is_file():
        return str(p)
    candidates = (
        [Path.cwd() / rel_or_abs]
        + [r / rel_or_abs for r in search_roots]
        + [r / "weights" / rel_or_abs for r in search_roots]
        + [r / Path(rel_or_abs).name for r in search_roots]
    )
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(p)


def _list_pt(*dirs: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for d in dirs:
        if d.is_dir():
            for p in sorted(d.glob("*.pt")):
                out.append({"value": p.name, "label": p.name, "path": str(p)})
    return out


def list_pose_models() -> list[dict[str, str]]:
    return _list_pt(paths.POSE_WEIGHTS_DIR, paths.PROJECT_ROOT / "weights")


def list_fire_models() -> list[dict[str, str]]:
    return _list_pt(paths.FIRE_WEIGHTS_DIR, paths.FIRE_WEIGHTS_DIR / "weights")


def _new_trackers(predictor) -> list:
    """Fresh tracker list for `predictor`, built by Ultralytics itself.

    on_predict_start(persist=False) always rebuilds `predictor.trackers` from
    the predictor's tracker config. Calling it (rather than re-implementing
    it) keeps this working across Ultralytics versions: 8.4 moved the YAML
    loader and added a re-ID hook for BoT-SORT.
    """
    from ultralytics.trackers.track import on_predict_start
    on_predict_start(predictor, persist=False)
    return predictor.trackers


# ---- Shared models --------------------------------------------------------

class Models:
    """Lazy-loaded container for pose / fire YOLO + face engine.

    `ensure_loaded()` is idempotent: it loads what is missing, unloads what was
    disabled, and reloads a model whose weights or precision setting changed.
    """

    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.RLock()
        # Serializes inference across per-camera worker threads. Ultralytics
        # YOLO predict()/track() share mutable predictor state and are NOT
        # thread-safe: concurrent calls from two cameras interleave, and one
        # camera can come back with detections computed from the OTHER
        # camera's frame (e.g. a fire on Camera 1 attributed to Camera 2).
        self.infer_lock = threading.Lock()
        self.pose_model = None
        self.fire_model = None
        self.face_engine = None
        self.device = self._resolve_device()
        self.half = False
        self.pose_weights_path: str = ""
        self.fire_weights_path: str = ""
        self._pose_key: Optional[tuple] = None
        self._fire_key: Optional[tuple] = None

    def ensure_loaded(self) -> None:
        with self._lock:
            # FP16 roughly halves GPU inference time with no practical change
            # in detections; it is a no-op on CPU.
            self.half = bool(self.settings.get("use_fp16", True)) and self.device != "cpu"
            pose_key = (self.settings.get("yolo_model", "yolo11n-pose.pt"), self.half)
            if self.pose_model is None or self._pose_key != pose_key:
                self._load_pose(pose_key)

            fire_key = (self.settings.get("fire_model", "best.pt"), self.half)
            if self.settings.get("fire_enabled", True):
                if self.fire_model is None or self._fire_key != fire_key:
                    self._load_fire(fire_key)
            elif self.fire_model is not None:
                log.info("fire detection disabled - unloading")
                self.fire_model, self._fire_key = None, None

            if self.settings.get("face_enabled", True):
                if self.face_engine is None:
                    self._load_face()
            elif self.face_engine is not None:
                log.info("face recognition disabled - unloading")
                self.face_engine = None

    def reload(self) -> None:
        with self._lock:
            self.pose_model = None
            self.fire_model = None
            self.face_engine = None
        self.ensure_loaded()

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"

    def _yolo(self, path: str):
        from ultralytics import YOLO
        model = YOLO(path)
        try:
            model.to(0 if self.device == "0" else self.device)
        except Exception as e:
            log.warning("could not move %s to device %s: %s", path, self.device, e)
        return model

    def _load_pose(self, key: tuple) -> None:
        weights = key[0]
        path = resolve_weights(weights, [
            paths.POSE_WEIGHTS_DIR, paths.PROJECT_ROOT, Path.cwd()
        ])
        if not Path(path).is_file():
            avail = list_pose_models()
            if avail:
                path = avail[0]["path"]
        log.info("loading pose: %s (fp16=%s)", path, self.half)
        self.pose_model = self._yolo(path)
        self.pose_weights_path = path
        self._pose_key = key

    def _load_fire(self, key: tuple) -> None:
        weights = key[0]
        path = resolve_weights(weights, [
            paths.FIRE_WEIGHTS_DIR, paths.PROJECT_ROOT, Path.cwd()
        ])
        if not Path(path).is_file():
            avail = list_fire_models()
            if avail:
                path = avail[0]["path"]
        if not Path(path).is_file():
            log.warning("fire weights not found at %s", weights)
            return
        log.info("loading fire: %s (fp16=%s)", path, self.half)
        self.fire_model = self._yolo(path)
        self.fire_weights_path = path
        self._fire_key = key

    def _load_face(self) -> None:
        try:
            self.face_engine = FaceEngine()
            if self.face_engine.available:
                log.info("face recognition: ready (InsightFace)")
            else:
                log.warning("face disabled: %s", self.face_engine.last_error)
        except Exception as e:
            log.exception("face engine init failed: %s", e)
            self.face_engine = None


# ---- Per-camera state -----------------------------------------------------

@dataclass
class FrameResult:
    ts: float
    persons: list[dict[str, Any]] = field(default_factory=list)
    fires: list[dict[str, Any]] = field(default_factory=list)
    faces: list[dict[str, Any]] = field(default_factory=list)
    events: list = field(default_factory=list)
    fps: float = 0.0
    fall_alert_ids: list[int] = field(default_factory=list)
    fire_alert: bool = False
    intruder_alert: bool = False
    danger_zones: list[dict[str, Any]] = field(default_factory=list)
    safe_zones: list[dict[str, Any]] = field(default_factory=list)


def _bbox4(b) -> Optional[tuple[float, float, float, float]]:
    arr = np.asarray(b).flatten().tolist()
    if len(arr) < 4:
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def associate_faces(persons: list[dict[str, Any]],
                    faces: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Map person track id -> recognised face lying in that person's head region.

    A face belongs to a person when its centre is inside the person's box,
    in the top half. Each face is given to at most one person (the one whose
    box is smallest, i.e. the tightest fit when people overlap).
    """
    out: dict[int, dict[str, Any]] = {}
    taken: set[int] = set()
    boxes = [(int(p["id"]), bb) for p in persons
             if (bb := _bbox4(p["bbox"])) is not None]
    boxes.sort(key=lambda it: (it[1][2] - it[1][0]) * (it[1][3] - it[1][1]))
    for pid, (x1, y1, x2, y2) in boxes:
        head_y2 = y1 + 0.5 * (y2 - y1)
        best, best_area = None, -1.0
        for i, f in enumerate(faces):
            if i in taken or f.get("match_id") is None:
                continue
            cx, cy = f["x"] + f["w"] / 2.0, f["y"] + f["h"] / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= head_y2 and f["w"] * f["h"] > best_area:
                best, best_area = i, f["w"] * f["h"]
        if best is not None:
            taken.add(best)
            out[pid] = faces[best]
    return out


# ---- FaceWorker -----------------------------------------------------------

class FaceWorker(threading.Thread):
    """
    Per-camera daemon thread that runs face detection + intruder logic
    OFF the main capture loop. The pipeline submits the latest frame via
    submit() (non-blocking, drops oldest if a frame is still pending) and
    reads back cached results via latest_faces() instantly.

    This keeps the main loop running at pose-only speed; face refreshes
    whenever this worker finishes a cycle.

    Intruder logic: only faces passing `is_good_face` are matched; an unknown
    face must persist for `intruder_confirm_frames` consecutive face cycles
    before the intruder alert is raised.
    """

    def __init__(self, *, camera_id, camera_name, models, settings,
                 person_store, intruder_store, bus):
        super().__init__(daemon=True, name=f"hs-face-{camera_id}")
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.models = models
        self.settings = settings
        self.person_store = person_store
        self.intruder_store = intruder_store
        self.bus = bus
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_faces: list[dict[str, Any]] = []
        self._intruder_active = False
        self._unknown_streak = 0
        self._intruder_emitted_at: float = 0.0
        self._stop_flag = threading.Event()

    def submit(self, frame_bgr: np.ndarray, ts: float) -> None:
        """Non-blocking. Drops the queued frame if one is still pending."""
        try:
            self._inbox.put_nowait((frame_bgr, ts))
        except queue.Full:
            try:
                self._inbox.get_nowait()
                self._inbox.put_nowait((frame_bgr, ts))
            except Exception:
                pass

    def latest_faces(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._latest_faces)

    def intruder_active(self) -> bool:
        with self._lock:
            return self._intruder_active

    def request_stop(self) -> None:
        self._stop_flag.set()

    def _publish(self, faces: list[dict[str, Any]], intruder: bool) -> None:
        with self._lock:
            self._latest_faces = faces
            self._intruder_active = intruder

    def run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                frame, ts = self._inbox.get(timeout=0.5)
            except queue.Empty:
                continue

            engine = self.models.face_engine
            if (not self.settings.get("face_enabled", True) or engine is None
                    or not engine.available or self.person_store is None):
                self._unknown_streak = 0
                self._publish([], False)
                continue

            try:
                with self.models.infer_lock:
                    faces = engine.detect(frame)
            except Exception as e:
                log.warning("face-worker cam=%s detect: %s", self.camera_id, e)
                continue

            gallery = self.person_store.gallery()
            match_threshold = float(self.settings.get("face_match_threshold", 0.45))
            min_size = float(self.settings.get("face_min_size", 40))
            min_score = float(self.settings.get("face_min_det_score", 0.6))
            for f in faces:
                f["match_id"] = f["match_name"] = f["match_category"] = None
                f["match_score"] = 0.0
                f["quality_ok"] = is_good_face(f, min_size=min_size,
                                               min_det_score=min_score)
                if not f["quality_ok"]:
                    continue
                pid, score = best_match(f["embedding"], gallery, threshold=match_threshold)
                f["match_id"] = pid
                f["match_score"] = score
                if pid is not None:
                    f["match_name"] = self.person_store.name_of(pid)
                    f["match_category"] = self.person_store.category_of(pid)

            unknown = [f for f in faces if f["quality_ok"] and f["match_id"] is None]
            self._unknown_streak = self._unknown_streak + 1 if unknown else 0
            confirm = max(1, int(self.settings.get("intruder_confirm_frames", 2)))
            intruder = self._unknown_streak >= confirm

            if intruder:
                cooldown = float(self.settings.get("intruder_cooldown", 30))
                if ts - self._intruder_emitted_at >= cooldown:
                    self._intruder_emitted_at = ts
                    self._emit_intruder(max(unknown, key=lambda x: x["w"] * x["h"]),
                                        frame, ts)

            self._publish(faces, intruder)

    def _emit_intruder(self, face: dict[str, Any], frame, ts: float) -> None:
        if self.intruder_store is not None:
            try:
                self.intruder_store.add(
                    embedding=face["embedding"],
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    frame_bgr=frame,
                    face_bbox=(face["x"], face["y"], face["w"], face["h"]),
                )
            except Exception as e:
                log.warning("face-worker intruder cam=%s: %s", self.camera_id, e)
        self.bus.publish(Event(
            event_type="intruder_detected", ts=ts,
            camera_id=self.camera_id, camera_name=self.camera_name,
            person_category="unknown",
            confidence=float(face.get("det_score", 0.5)),
            details="Unknown face on camera",
            bbox=(face["x"], face["y"],
                  face["x"] + face["w"], face["y"] + face["h"]),
            meta={"age": face.get("age")},
        ), frame=frame)


class CameraPipeline:
    """Per-camera mutable state. Models are external (shared)."""

    def __init__(self, *, camera_id: int, camera_name: str,
                 models: Models, settings, person_store=None,
                 zone_store=None, intruder_store=None, bus=None):
        self.camera_id = int(camera_id)
        self.camera_name = camera_name
        self.models = models
        self.settings = settings
        self.person_store = person_store
        self.zone_store = zone_store
        self.intruder_store = intruder_store

        self.cfg = Config(
            model_path=self.models.pose_weights_path or "yolo11n-pose.pt",
            device="auto",
            imgsz=int(self.settings.get("yolo_imgsz", 640)),
            conf=float(self.settings.get("yolo_confidence", 0.5)),
            tracker="bytetrack.yaml",
            show=False,
            save_path=None,
            inactivity_s=float(self.settings.get("inactivity_seconds", 300)),
        )
        self.fall_state = MultiPersonState(self.cfg)
        # This camera's own ByteTrack state (see _track()).
        self._trackers: Optional[list] = None
        self._pose_errors = 0
        # track id -> {"person_id", "name", "category"} once a face is recognised
        self._identity: dict[int, dict[str, Any]] = {}

        # cooldown bookkeeping
        self._prev_states: dict[int, Any] = {}
        self._fall_emitted: dict[int, float] = {}
        self._motionless_emitted: dict[int, float] = {}
        self._inactivity_emitted: dict[int, float] = {}
        self._zone_emitted: dict[tuple[int, int], float] = {}
        self._fire_emitted: dict[str, float] = {}
        self._fire_confirm = fire.FireConfirmer()

        # frame-skip caches so the overlay still has data on skipped frames
        self._frame_idx: int = 0
        self._cached_fires: list[dict[str, Any]] = []
        self._cached_fire_alert: bool = False

        self._smoothed_fps: float = 0.0
        self._last_t: Optional[float] = None

        # Async face worker (one per camera). Always running; it polls
        # face_enabled itself and sleeps when disabled.
        self.face_worker: Optional[FaceWorker] = None
        if bus is not None:
            try:
                self.face_worker = FaceWorker(
                    camera_id=self.camera_id, camera_name=self.camera_name,
                    models=self.models, settings=self.settings,
                    person_store=self.person_store,
                    intruder_store=self.intruder_store,
                    bus=bus,
                )
                self.face_worker.start()
            except Exception as e:
                log.warning("cam=%s face-worker start failed: %s", self.camera_id, e)
                self.face_worker = None

    def shutdown(self) -> None:
        if self.face_worker is not None:
            self.face_worker.request_stop()

    def reload_settings(self) -> None:
        self.cfg.imgsz = int(self.settings.get("yolo_imgsz", 640))
        self.cfg.conf = float(self.settings.get("yolo_confidence", 0.5))
        self.cfg.inactivity_s = float(self.settings.get("inactivity_seconds", 300))

    # ---- main entry point ----------------------------------------------

    def process(self, frame_bgr: np.ndarray,
                ts: Optional[float] = None) -> FrameResult:
        ts = ts if ts is not None else time.time()
        res = FrameResult(ts=ts)
        H, W = frame_bgr.shape[:2]
        device = 0 if self.models.device == "0" else self.models.device

        self._run_pose(frame_bgr, ts, H, W, device, res)
        self._run_face(frame_bgr, ts, res)
        self._attach_identities(res)
        self._collect_zones(W, H, res)
        self._emit_person_events(ts, W, H, res)
        # Suppress the red ALERT-FALL banner when fall detection is disabled,
        # even if the underlying FSM state still says FALL_DETECTED.
        if self.settings.get("fall_enabled", True):
            res.fall_alert_ids = [
                pid for pid, det in self.fall_state.detectors.items()
                if det.fall_alert
            ]
        else:
            res.fall_alert_ids = []
        self._run_fire(frame_bgr, ts, device, res)

        # FPS smoothing (EMA, alpha=0.1)
        if self._last_t is not None:
            dt = max(1e-6, ts - self._last_t)
            inst = 1.0 / dt
            self._smoothed_fps = (inst if self._smoothed_fps == 0.0
                                  else 0.9 * self._smoothed_fps + 0.1 * inst)
        self._last_t = ts
        res.fps = self._smoothed_fps
        self._frame_idx += 1
        return res

    # ---- stages ---------------------------------------------------------

    def _track(self, model, frame, device):
        """Pose + ByteTrack using this camera's own tracker state.

        Ultralytics keeps tracker state on the model's predictor, which all
        cameras share, so with two cameras each frame from one camera updated
        and expired the other camera's tracks: IDs churned every frame and the
        per-person fall FSMs never accumulated history. Swapping this camera's
        trackers in before each call (under infer_lock) isolates them.
        Never delete `predictor.trackers`: Model.track() would then register
        its tracking callbacks a second time.
        """
        pred = model.predictor
        if pred is not None and hasattr(pred, "trackers"):
            if self._trackers is None:
                self._trackers = _new_trackers(pred)
            pred.trackers = self._trackers
        out = model.track(
            frame,
            imgsz=self.cfg.imgsz, conf=self.cfg.conf,
            persist=True, tracker=self.cfg.tracker,
            verbose=False, device=device, half=self.models.half,
        )
        self._trackers = model.predictor.trackers
        return out

    def _run_pose(self, frame, ts, H, W, device, res: FrameResult) -> None:
        model = self.models.pose_model
        if model is None:
            return
        try:
            with self.models.infer_lock:
                tr = self._track(model, frame, device)
            if tr:
                res.persons = self.fall_state.step(tr[0], H, W, ts)
            self._pose_errors = 0
        except Exception as e:
            # A persistent error would otherwise log once per frame.
            self._pose_errors += 1
            if self._pose_errors <= 3 or self._pose_errors % 300 == 0:
                log.exception("pose cam=%s (error #%d): %s",
                              self.camera_id, self._pose_errors, e)

    def _attach_identities(self, res: FrameResult) -> None:
        """Remember who each track is once their face is recognised.

        The identity sticks to the track id for as long as the tracker keeps
        it, so a person stays known while facing away from the camera.
        """
        for pid, face in associate_faces(res.persons, res.faces).items():
            self._identity[pid] = {
                "person_id": face["match_id"],
                "name": face.get("match_name"),
                "category": face.get("match_category") or "unknown",
            }
        live = self.fall_state.detectors
        for pid in [p for p in self._identity if p not in live]:
            del self._identity[pid]
        for p in res.persons:
            ident = self._identity.get(int(p["id"]))
            p["name"] = ident["name"] if ident else None
            p["category"] = ident["category"] if ident else "unknown"

    def _collect_zones(self, W: int, H: int, res: FrameResult) -> None:
        if self.zone_store is None:
            return
        res.danger_zones = self.zone_store.danger_zones_for(self.camera_id, W, H)
        for sz in self.zone_store.for_camera(self.camera_id):
            if sz["zone_type"] == "safe":
                res.safe_zones.append({
                    **sz,
                    "polygon_scaled": scale_polygon(sz["polygon"], 640, 480, W, H),
                })

    def _emit_person_events(self, ts, W, H, res: FrameResult) -> None:
        cooldown = float(self.settings.get("alert_cooldown", 60))
        fall_thresh = float(self.settings.get("fall_threshold", 0.8))
        fall_enabled = bool(self.settings.get("fall_enabled", True))

        for p in res.persons:
            pid = int(p["id"])
            cur = p["detector"].state
            prev = self._prev_states.get(pid)
            self._prev_states[pid] = cur

            bb = _bbox4(p["bbox"])
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            foot = ((x1 + x2) / 2.0, y2)

            in_safe = any(point_in_polygon(foot[0], foot[1], z["polygon_scaled"])
                          for z in res.safe_zones)
            cat = p.get("category") or "unknown"
            who = f"Person #{pid}" + (f" ({p['name']})" if p.get("name") else "")

            def _emit(emit_dict, event_type, conf, details, extra_meta=None):
                last = emit_dict.get(pid, 0.0)
                if (ts - last) < cooldown:
                    return
                emit_dict[pid] = ts
                meta = {"person_id": pid, "fsm_state": cur.value}
                if p.get("name"):
                    meta["name"] = p["name"]
                if extra_meta:
                    meta.update(extra_meta)
                res.events.append(Event(
                    event_type=event_type, ts=ts,
                    camera_id=self.camera_id, camera_name=self.camera_name,
                    person_category=cat,
                    confidence=conf, details=details,
                    bbox=bb, meta=meta,
                ))

            # rising-edge events; suppressed inside safe zones AND when
            # fall detection has been disabled in the dashboard.
            if fall_enabled and not in_safe:
                if cur == State.FALL_DETECTED and prev != State.FALL_DETECTED:
                    _emit(self._fall_emitted, "fall_detected", fall_thresh, who)
                if cur == State.LYING_MOTIONLESS and prev != State.LYING_MOTIONLESS:
                    _emit(self._motionless_emitted, "lying_motionless",
                          0.95, f"{who} unresponsive")
                if cur == State.INACTIVITY and prev != State.INACTIVITY:
                    _emit(self._inactivity_emitted, "inactivity",
                          0.7, f"{who} idle")

            # Child-in-danger-zone (uses a separate cooldown dict keyed by pid+zone)
            if cat == "child":
                for zone in res.danger_zones:
                    if not point_in_polygon(foot[0], foot[1], zone["polygon_scaled"]):
                        continue
                    key = (pid, zone["zone_id"])
                    last = self._zone_emitted.get(key, 0.0)
                    if (ts - last) < cooldown:
                        continue
                    self._zone_emitted[key] = ts
                    res.events.append(Event(
                        event_type="zone_entry", ts=ts,
                        camera_id=self.camera_id, camera_name=self.camera_name,
                        person_category=cat, confidence=0.9,
                        details=f"Child {p.get('name') or f'#{pid}'} entered "
                                f"{zone['zone_name']}",
                        bbox=bb,
                        meta={"person_id": pid, "zone_id": zone["zone_id"]},
                    ))

        # Forget per-track bookkeeping for tracks the FSM has dropped.
        live = self.fall_state.detectors
        for d in (self._prev_states, self._fall_emitted,
                  self._motionless_emitted, self._inactivity_emitted):
            for pid in [k for k in d if k not in live]:
                del d[pid]

    def _run_fire(self, frame, ts, device, res: FrameResult) -> None:
        if not self.settings.get("fire_enabled", True):
            return
        every = max(1, int(self.settings.get("fire_every_n", 2) or 1))
        if self._frame_idx % every != 0 or self.models.fire_model is None:
            res.fires = list(self._cached_fires)
            res.fire_alert = self._cached_fire_alert
            return
        try:
            with self.models.infer_lock:
                res.fires = fire.predict(
                    self.models.fire_model, frame,
                    conf=float(self.settings.get("fire_confidence", 0.35)),
                    imgsz=self.cfg.imgsz, device=device, half=self.models.half,
                )
        except Exception as e:
            log.warning("fire cam=%s: %s", self.camera_id, e)
            return
        alert_classes = {
            c.strip().lower()
            for c in str(self.settings.get("fire_classes", "fire,smoke")).split(",")
            if c.strip()
        }
        per_cls_best: dict[str, tuple[float, tuple]] = {}
        for d in res.fires:
            if d["cls_name"] not in alert_classes:
                continue
            best = per_cls_best.get(d["cls_name"])
            if best is None or d["conf"] > best[0]:
                per_cls_best[d["cls_name"]] = (d["conf"], d["bbox"])

        confirmed = self._fire_confirm.update(
            set(per_cls_best),
            k=int(self.settings.get("fire_confirm_frames", 3)),
            n=int(self.settings.get("fire_confirm_window", 5)),
        )
        res.fire_alert = bool(confirmed)
        cooldown = float(self.settings.get("fire_cooldown", 5))
        for cls_name in confirmed & set(per_cls_best):
            if ts - self._fire_emitted.get(cls_name, 0.0) < cooldown:
                continue
            self._fire_emitted[cls_name] = ts
            best_conf, bbox = per_cls_best[cls_name]
            res.events.append(Event(
                event_type="fire_detected", ts=ts,
                camera_id=self.camera_id, camera_name=self.camera_name,
                person_category="unknown",
                confidence=best_conf,
                details=cls_name.upper(),
                bbox=bbox,
                meta={"class": cls_name},
            ))
        self._cached_fires = list(res.fires)
        self._cached_fire_alert = res.fire_alert

    def _run_face(self, frame, ts, res: FrameResult) -> None:
        """Hand the frame to the async FaceWorker; read its latest result."""
        if self.face_worker is None:
            return
        if not self.settings.get("face_enabled", True):
            res.faces = []
            return
        every = max(1, int(self.settings.get("face_every_n", 5) or 1))
        if self._frame_idx % every == 0:
            self.face_worker.submit(frame, ts)
        res.faces = self.face_worker.latest_faces()
        res.intruder_alert = self.face_worker.intruder_active()
