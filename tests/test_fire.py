import threading
from types import SimpleNamespace

import numpy as np

from homeshield.detectors.fire import FireConfirmer, parse_result
from homeshield.pipeline import CameraPipeline


class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _Boxes:
    def __init__(self, a):
        self.xyxy, self.conf, self.cls = _T(a[:, :4]), _T(a[:, 4]), _T(a[:, 5])
        self._n = len(a)

    def __len__(self):
        return self._n


NAMES = {0: "Fire", 1: "other", 2: "smoke"}


def _result(dets):
    """Ultralytics-like detect result from [(x1, y1, x2, y2, conf, cls), ...]."""
    boxes = _Boxes(np.asarray(dets, dtype=np.float32)) if dets else None
    return SimpleNamespace(boxes=boxes, names=NAMES)


def test_parse_result_lowercases_class_names():
    out = parse_result(_result([(1, 2, 3, 4, 0.8, 0), (5, 6, 7, 8, 0.4, 2)]))
    assert out == [
        {"bbox": (1.0, 2.0, 3.0, 4.0), "conf": np.float32(0.8).item(), "cls_name": "fire"},
        {"bbox": (5.0, 6.0, 7.0, 8.0), "conf": np.float32(0.4).item(), "cls_name": "smoke"},
    ]
    assert parse_result(_result([])) == []


def test_confirmer_ignores_single_frame_blips():
    c = FireConfirmer()
    seen = [c.update({"fire"} if i in (0, 4, 8) else set(), k=3, n=5) for i in range(10)]
    assert not any(seen)


def test_confirmer_confirms_persistent_fire_and_clears_after():
    c = FireConfirmer()
    on = [c.update({"fire"}, k=3, n=5) for _ in range(4)]
    assert on == [set(), set(), {"fire"}, {"fire"}]
    off = [c.update(set(), k=3, n=5) for _ in range(5)]
    assert off[-1] == set()


class _FakeFireModel:
    def __init__(self, dets): self.dets = dets
    def predict(self, frame, **kw): return [_result(self.dets)]


def _pipeline(fire_model, **settings):
    models = SimpleNamespace(pose_model=None, fire_model=fire_model, face_engine=None,
                             device="cpu", half=False, pose_weights_path="",
                             infer_lock=threading.Lock())
    s = {"fire_every_n": 1, "fire_cooldown": 5, **settings}
    return CameraPipeline(camera_id=1, camera_name="t", models=models, settings=s)


def test_pipeline_fire_event_needs_confirmation_and_respects_cooldown():
    pipe = _pipeline(_FakeFireModel([(10, 10, 50, 50, 0.9, 0)]))
    frame = np.zeros((480, 640, 3), np.uint8)
    results = [pipe.process(frame, ts=100.0 + i * 0.1) for i in range(8)]
    alerts = [r.fire_alert for r in results]
    events = [[e.event_type for e in r.events] for r in results]
    assert alerts == [False, False] + [True] * 6        # banner on 3rd run, stays on
    assert events[2] == ["fire_detected"]               # one event ...
    assert sum(len(e) for e in events) == 1             # ... then cooldown
    assert results[0].fires                             # boxes drawn immediately


def test_pipeline_ignores_non_alert_classes():
    pipe = _pipeline(_FakeFireModel([(10, 10, 50, 50, 0.9, 1)]))   # "other"
    frame = np.zeros((480, 640, 3), np.uint8)
    rs = [pipe.process(frame, ts=float(i)) for i in range(6)]
    assert not any(r.fire_alert or r.events for r in rs)
