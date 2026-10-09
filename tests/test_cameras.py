import threading
import time

import cv2
import numpy as np

from homeshield.cameras import FrameGrabber, LatestFrame


def test_latest_frame_encodes_lazily_once_per_frame(monkeypatch):
    calls = []
    real = cv2.imencode
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: calls.append(1) or real(*a, **k))
    lf = LatestFrame()
    clean = np.zeros((48, 64, 3), np.uint8)
    shown = np.full((48, 64, 3), 255, np.uint8)
    lf.set(shown, clean)
    assert calls == []                       # nothing encoded until someone asks
    j1 = lf.jpeg()
    j2, _ = lf.get_blocking(last_version=0)
    assert j1 and j1 == j2 and len(calls) == 1
    assert cv2.imdecode(np.frombuffer(j1, np.uint8), 1).mean() > 200   # annotated frame
    assert lf.raw().max() == 0                                         # clean frame


def test_latest_frame_get_blocking_wakes_on_new_frame():
    lf = LatestFrame()
    lf.set(np.zeros((8, 8, 3), np.uint8))
    threading.Timer(0.05, lambda: lf.set(np.ones((8, 8, 3), np.uint8))).start()
    t = time.perf_counter()
    _, version = lf.get_blocking(last_version=1, timeout=2.0)
    assert version == 2 and time.perf_counter() - t < 1.0


class _BufferedStream:
    """A 50 FPS network stream whose frames queue up until read (like FFmpeg).
    Each 'frame' carries its own capture time."""

    def __init__(self, fps=50.0):
        self.dt, self.t0, self.n = 1.0 / fps, time.time(), 0
        self.released = False

    def read(self):
        due = self.t0 + self.n * self.dt
        if due > time.time():
            time.sleep(due - time.time())
        self.n += 1
        return True, np.array([[due]])

    def release(self):
        self.released = True


def test_sequential_reads_fall_behind_a_buffered_stream():
    """The failure mode the grabber exists for: 80 ms inference, 50 FPS stream."""
    cap, ages = _BufferedStream(), []
    for _ in range(12):
        _, frame = cap.read()
        ages.append(time.time() - frame[0, 0])
        time.sleep(0.08)
    assert ages[-1] > 0.4                  # half a second behind and growing


def test_grabber_keeps_latency_bounded_when_inference_is_slow():
    cap = _BufferedStream()
    g = FrameGrabber(cap, name="test-grab", max_fails=5)
    g.start()
    version, ages = 0, []
    try:
        for _ in range(12):
            item = g.next(version, timeout=1.0)
            assert item is not None
            frame, ts, version = item
            ages.append(time.time() - frame[0, 0])
            time.sleep(0.08)                                    # slow inference
    finally:
        g.stop()
        g.join(timeout=2.0)
    assert max(ages[3:]) < 0.06            # never more than ~2 stream frames old
    assert cap.released                    # grabber released the capture


def test_camera_source_parsing_strips_copy_as_path_quotes(tmp_path):
    from homeshield.cameras import _is_file_source, _parse_source
    video = tmp_path / "clip with spaces.mp4"
    video.write_bytes(b"")
    assert _parse_source(f'"{video}"') == str(video)
    assert _parse_source(f"  '{video}' ") == str(video)
    assert _is_file_source(_parse_source(f'"{video}"'))
    assert _parse_source(" 0 ") == 0
    assert _parse_source('"1"') == 1
    assert _parse_source("rtsp://cam/stream") == "rtsp://cam/stream"


def test_latest_frame_reports_camera_size_not_placeholder_size():
    lf = LatestFrame()
    assert lf.size() is None
    cam = np.zeros((720, 1280, 3), np.uint8)
    lf.set(cam, cam)
    assert lf.size() == (1280, 720)
    lf.set(np.zeros((540, 960, 3), np.uint8))      # offline placeholder, no raw frame
    assert lf.size() == (1280, 720)
