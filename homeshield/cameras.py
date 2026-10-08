"""Multi-camera lifecycle: CameraStore, LatestFrame, FrameGrabber,
CaptureWorker, CameraManager."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .annotator import annotate, disconnected_placeholder
from .db import read_conn, write_conn
from .events import Event, EventBus
from .pipeline import CameraPipeline, Models

log = logging.getLogger(__name__)


# ---- CameraStore ----------------------------------------------------------

class CameraStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def list(self) -> list[dict[str, Any]]:
        with read_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM cameras ORDER BY camera_id ASC"
            ).fetchall()
        return [{
            "camera_id": r["camera_id"],
            "name": r["name"],
            "url": r["url"],
            "location": r["location"] or "",
            "enabled": bool(r["enabled"]),
            "created_at": r["created_at"],
        } for r in rows]

    def add(self, *, name: str, url: str, location: str = "") -> int:
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO cameras (name, url, location) VALUES (?, ?, ?)",
                (name.strip() or "Camera",
                 url.strip() or "0",
                 location.strip()),
            )
            return cur.lastrowid

    def delete(self, camera_id: int) -> None:
        with write_conn(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cameras WHERE camera_id = ?", (int(camera_id),)
            )


# ---- LatestFrame ----------------------------------------------------------

class LatestFrame:
    """Thread-safe single slot: the annotated frame shown to viewers plus the
    clean camera frame it was drawn from.

    JPEG encoding is lazy: it happens once per new frame, only when a viewer
    asks for it, on the viewer's thread. Previously every frame of every
    camera was encoded on the capture thread even with nobody watching.

    `raw()` returns the CLEAN frame. Face enrolment reads it, and running
    ArcFace on the annotated frame (skeleton dots on the eyes and nose, a box
    and label around the face) produced degraded enrolment embeddings.
    """

    def __init__(self, jpeg_quality: int = 80):
        self._cond = threading.Condition()
        self._quality = int(jpeg_quality)
        self._display: Optional[np.ndarray] = None
        self._raw: Optional[np.ndarray] = None
        self._version = 0
        self._jpeg: Optional[bytes] = None
        self._jpeg_version = -1

    def set(self, display: np.ndarray, raw: Optional[np.ndarray] = None) -> None:
        """Publish a new frame. Takes ownership: callers must not modify either
        array afterwards."""
        with self._cond:
            self._display = display
            self._raw = raw
            self._version += 1
            self._cond.notify_all()

    def _encoded(self) -> tuple[Optional[bytes], int]:
        with self._cond:
            img, ver = self._display, self._version
            if self._jpeg_version == ver:
                return self._jpeg, ver
        if img is None:
            return None, ver
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        jpeg = buf.tobytes() if ok else None
        with self._cond:
            if ver > self._jpeg_version:
                self._jpeg, self._jpeg_version = jpeg, ver
        return jpeg, ver

    def get_blocking(self, last_version: int, timeout: float = 1.0):
        with self._cond:
            if self._version == last_version:
                self._cond.wait(timeout=timeout)
        return self._encoded()

    def jpeg(self) -> Optional[bytes]:
        return self._encoded()[0]

    def raw(self) -> Optional[np.ndarray]:
        with self._cond:
            return None if self._raw is None else self._raw.copy()


# ---- FrameGrabber ---------------------------------------------------------

class FrameGrabber(threading.Thread):
    """Reads a live source continuously, keeping only the newest frame.

    Network streams (RTSP / HTTP) buffer inside FFmpeg, where
    CAP_PROP_BUFFERSIZE has no effect: when inference is slower than the
    stream, a read-then-infer loop drifts further and further behind real
    time and alerts arrive late. Reading on its own thread always hands the
    pipeline the newest frame, stamps it with its capture time, and overlaps
    camera I/O with inference.

    The grabber owns the capture and releases it when it stops.
    """

    def __init__(self, cap: cv2.VideoCapture, *, name: str, max_fails: int):
        super().__init__(daemon=True, name=name)
        self._cap = cap
        self._max_fails = max_fails
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._version = 0
        self.done = False
        self.read_fails = 0
        self._stop_flag = threading.Event()   # not `_stop`: see CaptureWorker

    def run(self) -> None:
        try:
            while not self._stop_flag.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self.read_fails += 1
                    if self.read_fails >= self._max_fails:
                        break
                    self._stop_flag.wait(0.05)
                    continue
                self.read_fails = 0
                with self._cond:
                    self._frame, self._ts = frame, time.time()
                    self._version += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self.done = True
                self._cond.notify_all()
            self._cap.release()

    def next(self, last_version: int, timeout: float = 1.0):
        """Newest (frame, capture_ts, version) newer than `last_version`, or
        None on timeout / once the grabber has stopped."""
        with self._cond:
            if self._version == last_version and not self.done:
                self._cond.wait(timeout)
            if self._version == last_version:
                return None
            return self._frame, self._ts, self._version

    def stop(self) -> None:
        self._stop_flag.set()


# ---- CaptureWorker --------------------------------------------------------

@dataclass
class WorkerStatus:
    started_at: float = 0.0
    frames_total: int = 0
    last_error: Optional[str] = None
    camera_connected: bool = False
    fps: float = 0.0


def _parse_source(s):
    if isinstance(s, str):
        s = s.strip()
        if s.isdigit():
            return int(s)
    return s


def _is_file_source(src) -> bool:
    return isinstance(src, str) and Path(src).is_file()


class _PipelineFailed(Exception):
    """Too many consecutive pipeline errors: reconnect the camera."""


class CaptureWorker(threading.Thread):
    """
    NOTE: do NOT name the stop flag ``self._stop`` -- it shadows
    ``threading.Thread._stop`` (called by Thread.join) and crashes.
    """

    MAX_READ_FAILS = 30   # ~1.5 s of failed reads, survive transient hiccups
    MAX_PIPE_ERRORS = 60  # consecutive pipeline errors -> reconnect

    def __init__(self, camera: dict, pipeline: CameraPipeline,
                 bus: EventBus, latest: LatestFrame):
        super().__init__(daemon=True, name=f"hs-cam-{camera['camera_id']}")
        self.camera = camera
        self.pipeline = pipeline
        self.bus = bus
        self.latest = latest
        self.status = WorkerStatus()
        self._stop_flag = threading.Event()
        self._pipe_errors = 0
        self._last_ts = 0.0

    def request_stop(self):
        self._stop_flag.set()

    # ---- camera open ----------------------------------------------------

    def _open(self):
        src = _parse_source(self.camera["url"])
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Windows fallback
            except Exception:
                pass
            if not cap.isOpened():
                return None
        # Keep buffer tiny so inference lag doesn't pile up stale frames.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    # ---- main loop ------------------------------------------------------

    def run(self):
        self.status.started_at = time.time()
        backoff = 1.0
        cname = self.camera["name"]
        is_file = _is_file_source(_parse_source(self.camera["url"]))

        while not self._stop_flag.is_set():
            cap = self._open()
            if cap is None:
                if self.status.camera_connected:
                    self._sys_event(f"Camera disconnected ({self.camera['url']})", 0.0)
                self.status.camera_connected = False
                self.status.last_error = f"Could not open {self.camera['url']}"
                self.latest.set(disconnected_placeholder(text=f"{cname}: offline"))
                if self._stop_flag.wait(timeout=backoff):
                    break
                backoff = min(10.0, backoff * 1.7)
                continue

            if not self.status.camera_connected:
                self._sys_event(f"Camera connected ({self.camera['url']})", 1.0)
            self.status.camera_connected = True
            self.status.last_error = None
            backoff = 1.0
            self._pipe_errors = 0

            try:
                if is_file:
                    self._loop_file(cap)
                else:
                    self._loop_live(cap)
            except Exception as e:
                self.status.last_error = repr(e)
                self._sys_event(f"Pipeline error: {e}", 0.0)
            finally:
                self.status.camera_connected = False

    def _sys_event(self, msg: str, conf: float) -> None:
        self.bus.publish(Event(
            event_type="system",
            camera_id=self.camera["camera_id"],
            camera_name=self.camera["name"],
            details=msg,
            confidence=conf,
        ))

    def _loop_live(self, cap: cv2.VideoCapture) -> None:
        """Webcams and network streams: always process the newest frame."""
        grabber = FrameGrabber(cap, name=f"hs-grab-{self.camera['camera_id']}",
                               max_fails=self.MAX_READ_FAILS)
        grabber.start()
        version = 0
        try:
            while not self._stop_flag.is_set():
                item = grabber.next(version, timeout=1.0)
                if item is None:
                    if grabber.done:
                        return      # source failed: run() reconnects
                    self.status.last_error = (
                        f"No new frame ({grabber.read_fails}/{self.MAX_READ_FAILS} "
                        "failed reads)")
                    continue
                frame, ts, version = item
                loop_start = time.time()
                self._handle(frame, ts)
                self._throttle(loop_start)
        finally:
            grabber.stop()      # the grabber releases the capture itself
            # Give it a moment to do so, so an immediate reconnect to the same
            # device (e.g. webcam 0) doesn't find it still open.
            grabber.join(timeout=2.0)

    def _loop_file(self, cap: cv2.VideoCapture) -> None:
        """Video files: every frame, timestamped by the file's own clock and
        played no faster than real time, looping at the end.

        Wall-clock timestamps would make a 30 FPS clip processed at 10 FPS look
        3x slower to the fall FSM (every velocity 3x too small).
        """
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1.0 <= fps <= 240.0 else 30.0
            base = max(time.time(), self._last_ts + 1.0 / fps)
            idx = 0
            while not self._stop_flag.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if idx == 0:
                        return      # unreadable file: run() retries with backoff
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    base, idx = self._last_ts + 1.0 / fps, 0
                    continue
                ts = base + idx / fps
                idx += 1
                ahead = ts - time.time()
                if ahead > 0 and self._stop_flag.wait(timeout=ahead):
                    return
                loop_start = time.time()
                self._handle(frame, ts)
                self._throttle(loop_start)
        finally:
            cap.release()

    def _throttle(self, loop_start: float) -> None:
        """process_fps==0 -> unlimited; >0 -> soft cap."""
        max_fps = int(self.pipeline.settings.get("process_fps", 0) or 0)
        if max_fps > 0:
            slack = (1.0 / max_fps) - (time.time() - loop_start)
            if slack > 0:
                self._stop_flag.wait(timeout=slack)

    def _handle(self, frame: np.ndarray, ts: float) -> None:
        cid = self.camera["camera_id"]
        self._last_ts = ts
        try:
            res = self.pipeline.process(frame, ts=ts)
            self._pipe_errors = 0
        except Exception as e:
            self._pipe_errors += 1
            self.status.last_error = repr(e)
            if self._pipe_errors <= 3 or self._pipe_errors % 30 == 0:
                log.exception("cam=%s frame error #%d: %s", cid, self._pipe_errors, e)
            if self._pipe_errors >= self.MAX_PIPE_ERRORS:
                raise _PipelineFailed(repr(e)) from e
            self.latest.set(frame, frame)
            return

        for ev in res.events:
            try:
                self.bus.publish(ev, frame=frame)
            except Exception as e:
                log.warning("cam=%s event publish error: %s", cid, e)

        try:
            annotated = annotate(frame.copy(), res, camera_name=self.camera["name"])
        except Exception as e:
            log.exception("cam=%s annotate error: %s", cid, e)
            annotated = frame
        self.latest.set(annotated, frame)
        self.status.frames_total += 1
        self.status.fps = res.fps


# ---- CameraManager --------------------------------------------------------

class CameraManager:
    def __init__(self, *, db_path: str, models: Models, settings,
                 bus: EventBus, person_store, intruder_store, zone_store):
        self.db_path = db_path
        self.models = models
        self.settings = settings
        self.bus = bus
        self.person_store = person_store
        self.intruder_store = intruder_store
        self.zone_store = zone_store
        self.store = CameraStore(db_path)

        self._workers: dict[int, CaptureWorker] = {}
        self._frames: dict[int, LatestFrame] = {}
        self._pipelines: dict[int, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._running = False

    # ---- public API -----------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def latest(self, camera_id: int) -> Optional[LatestFrame]:
        with self._lock:
            return self._frames.get(int(camera_id))

    def status(self) -> dict[str, Any]:
        cams = self.store.list()
        out = {
            "running": self._running,
            "cameras_total": len(cams),
            "cameras_online": 0,
            "people_count": 0,
            "alerts_today": self.bus.count_today(),
            "cameras": {},
        }
        for c in cams:
            cid = c["camera_id"]
            w = self._workers.get(cid)
            pipe = self._pipelines.get(cid)
            active = bool(w and w.status.camera_connected)
            cam_people = (len(pipe.fall_state.detectors) if pipe else 0)
            if active:
                out["cameras_online"] += 1
            out["people_count"] += cam_people
            out["cameras"][str(cid)] = {
                "name": c["name"],
                "location": c["location"],
                "url": c["url"],
                "active": active,
                "fps": round(w.status.fps, 1) if w else 0.0,
                "people": cam_people,
            }
        return out

    # ---- start / stop ---------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self.models.ensure_loaded()
            for c in self.store.list():
                if c["enabled"]:
                    self._spawn_locked(c)
            self._running = True

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            pipes = list(self._pipelines.values())
            self._workers.clear()
            self._frames.clear()
            self._pipelines.clear()
            self._running = False
        # Stop face workers first (cheap), then capture workers.
        for p in pipes:
            try:
                p.shutdown()
            except Exception:
                pass
        # Join outside the lock so a worker stuck on cap.read() doesn't deadlock.
        for w in workers:
            w.request_stop()
        for w in workers:
            try:
                w.join(timeout=3.0)
            except Exception as e:
                log.warning("worker join failed: %s", e)

    # ---- live (re)config ------------------------------------------------

    def add_camera(self, *, name: str, url: str, location: str = "") -> int:
        cid = self.store.add(name=name, url=url, location=location)
        if self._running:
            cam = next((c for c in self.store.list() if c["camera_id"] == cid), None)
            if cam is not None:
                with self._lock:
                    self._spawn_locked(cam)
        return cid

    def delete_camera(self, camera_id: int) -> None:
        with self._lock:
            w = self._workers.pop(int(camera_id), None)
            self._frames.pop(int(camera_id), None)
            p = self._pipelines.pop(int(camera_id), None)
        if p is not None:
            try:
                p.shutdown()
            except Exception:
                pass
        if w is not None:
            w.request_stop()
            try:
                w.join(timeout=3.0)
            except Exception as e:
                log.warning("camera %s join failed: %s", camera_id, e)
        self.store.delete(camera_id)

    def reload_settings(self) -> None:
        # Apply live-tunable settings on each per-camera pipeline.
        with self._lock:
            for pipe in self._pipelines.values():
                pipe.reload_settings()
        # Pick up fire_enabled / face_enabled toggles without restarting cameras.
        # ensure_loaded() loads if enabled-and-missing and unloads if disabled.
        if self._running:
            try:
                self.models.ensure_loaded()
            except Exception as e:
                log.exception("ensure_loaded during reload failed: %s", e)

    # ---- internals ------------------------------------------------------

    def _spawn_locked(self, cam):
        cid = cam["camera_id"]
        if cid in self._workers:
            return
        pipe = CameraPipeline(
            camera_id=cid, camera_name=cam["name"],
            models=self.models, settings=self.settings,
            person_store=self.person_store,
            zone_store=self.zone_store,
            intruder_store=self.intruder_store,
            bus=self.bus,
        )
        latest = LatestFrame()
        latest.set(disconnected_placeholder(text=f"{cam['name']}: starting..."))
        worker = CaptureWorker(camera=cam, pipeline=pipe,
                               bus=self.bus, latest=latest)
        self._pipelines[cid] = pipe
        self._frames[cid] = latest
        self._workers[cid] = worker
        worker.start()
