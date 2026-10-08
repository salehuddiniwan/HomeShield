"""CameraPipeline behaviour with stand-in models (no GPU or weights needed)."""

import threading
import time
from types import SimpleNamespace

import numpy as np

import homeshield.pipeline as pipeline_mod
from homeshield.pipeline import CameraPipeline, FaceWorker


# ---- per-camera tracker isolation -----------------------------------------

class _Tracker:
    def __init__(self):
        self.cameras_seen = []


class _Predictor:
    pass


class _PoseModel:
    """Mimics how Ultralytics stores tracker state on the shared predictor."""

    def __init__(self):
        self.predictor = None

    def track(self, frame, **kw):
        if self.predictor is None:
            self.predictor = _Predictor()
            self.predictor.trackers = [_Tracker()]
        self.predictor.trackers[0].cameras_seen.append(int(frame[0, 0, 0]))
        return [SimpleNamespace(boxes=None, keypoints=None)]


def _models(pose=None, face=None):
    return SimpleNamespace(pose_model=pose, fire_model=None, face_engine=face,
                           device="cpu", half=False, pose_weights_path="",
                           infer_lock=threading.Lock())


def test_each_camera_gets_its_own_tracker(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "_new_trackers", lambda predictor: [_Tracker()])
    models = _models(pose=_PoseModel())
    settings = {"fire_enabled": False}
    cams = {cid: CameraPipeline(camera_id=cid, camera_name=str(cid),
                                models=models, settings=settings) for cid in (1, 2, 3)}
    for _ in range(4):
        for cid, pipe in cams.items():
            pipe.process(np.full((8, 8, 3), cid, np.uint8), ts=time.time())
    for cid, pipe in cams.items():
        assert pipe._trackers[0].cameras_seen == [cid] * 4


# ---- face worker: quality gate + intruder confirmation --------------------

class _Engine:
    available = True

    def __init__(self, faces):
        self.faces = faces

    def detect(self, frame):
        return [dict(f) for f in self.faces]


class _People:
    def gallery(self): return [(1, np.eye(512, dtype=np.float32)[0])]
    def name_of(self, pid): return "Alice"
    def category_of(self, pid): return "elderly"


class _Bus:
    def __init__(self): self.events = []
    def publish(self, ev, frame=None): self.events.append(ev)


def _face(axis, **kw):
    f = {"x": 10.0, "y": 10.0, "w": 80.0, "h": 90.0, "det_score": 0.9, "age": 30,
         "embedding": np.eye(512, dtype=np.float32)[axis]}
    f.update(kw)
    return f


def _run_worker(faces, cycles, **settings):
    bus = _Bus()
    w = FaceWorker(camera_id=1, camera_name="t", models=_models(face=_Engine(faces)),
                   settings={"intruder_confirm_frames": 2, **settings},
                   person_store=_People(), intruder_store=None, bus=bus)
    w.start()
    states = []
    try:
        for i in range(cycles):
            w.submit(np.zeros((8, 8, 3), np.uint8), ts=100.0 + i)
            deadline = time.time() + 2.0
            while w._inbox.qsize() and time.time() < deadline:
                time.sleep(0.005)
            time.sleep(0.05)
            states.append(w.intruder_active())
    finally:
        w.request_stop()
    return states, bus.events, w.latest_faces()


def test_unknown_face_needs_two_cycles_before_intruder_alert():
    states, events, _ = _run_worker([_face(axis=5)], cycles=3)
    assert states == [False, True, True]
    assert [e.event_type for e in events] == ["intruder_detected"]


def test_known_face_is_named_and_never_an_intruder():
    states, events, faces = _run_worker([_face(axis=0)], cycles=3)
    assert not any(states) and not events
    assert faces[0]["match_name"] == "Alice"


def test_small_or_unsure_faces_are_not_intruders():
    states, events, faces = _run_worker(
        [_face(axis=5, w=20.0, h=24.0), _face(axis=6, det_score=0.3)], cycles=3)
    assert not any(states) and not events
    assert [f["quality_ok"] for f in faces] == [False, False]
