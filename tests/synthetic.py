"""Synthetic COCO-17 skeleton tracks for exercising the fall FSM.

Each scenario is a function of time returning (hip_x, hip_y, trunk_angle_deg,
seated_blend). `skeleton()` turns that into 17 keypoints, and `run()` feeds the
sequence at a chosen frame rate through a MultiPersonState via a minimal stand-in
for an Ultralytics tracking result.

These tracks are deliberately simple (rigid limbs, Gaussian jitter, random
keypoint dropouts). They check the FSM's logic and frame-rate robustness; they
are not a substitute for evaluation on real fall videos.
"""

from __future__ import annotations

import numpy as np

# Body dimensions in pixels (person ~180 px tall in a 640x480 frame).
TORSO, HEAD, SHOULDER_W, HIP_W = 60.0, 25.0, 20.0, 12.0
THIGH, SHIN, UPPER_ARM, FOREARM = 45.0, 45.0, 30.0, 28.0
FLOOR_Y, STAND_HIP_Y = 440.0, 350.0


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u                       # accelerating, like a body under gravity


def _lerp(a, b, u):
    return a + (b - a) * min(max(u, 0.0), 1.0)


def skeleton(hx, hy, theta_deg, seated=0.0) -> np.ndarray:
    """17x2 keypoints. theta = trunk lean from vertical (90 = lying flat)."""
    t = np.radians(theta_deg)
    up = np.array([np.sin(t), -np.cos(t)])        # hip -> shoulder direction
    side = np.array([np.cos(t), np.sin(t)])        # across the body
    hip = np.array([hx, hy])
    sh = hip + TORSO * up
    nose = sh + HEAD * up
    k = {}
    k[0] = nose
    k[1], k[2] = nose + 4 * up - 5 * side, nose + 4 * up + 5 * side
    k[3], k[4] = nose - 9 * side, nose + 9 * side
    k[5], k[6] = sh - SHOULDER_W * side, sh + SHOULDER_W * side
    k[7], k[8] = k[5] - UPPER_ARM * up, k[6] - UPPER_ARM * up
    k[9], k[10] = k[7] - FOREARM * up, k[8] - FOREARM * up
    k[11], k[12] = hip - HIP_W * side, hip + HIP_W * side
    for h_i, kn_i, an_i in ((11, 13, 15), (12, 14, 16)):
        straight_knee = k[h_i] - THIGH * up
        straight_ankle = straight_knee - SHIN * up
        seated_knee = k[h_i] + np.array([THIGH, 0.0])
        seated_ankle = seated_knee + np.array([0.0, SHIN])
        k[kn_i] = _lerp(straight_knee, seated_knee, seated)
        k[an_i] = _lerp(straight_ankle, seated_ankle, seated)
    return np.stack([k[i] for i in range(17)])


# ---- scenarios: t -> (hip_x, hip_y, theta, seated) ------------------------

LYING_HIP_Y = FLOOR_Y - 15.0


def fall(t, start=3.0, dur=0.6):
    u = _ease((t - start) / dur)
    return 320 + 40 * u, _lerp(STAND_HIP_Y, LYING_HIP_Y, u), 88 * u, 0.0


def fast_fall(t):
    return fall(t, dur=0.4)


def slow_fall(t):
    return fall(t, dur=0.9)


def sit_down(t, start=3.0, dur=0.8):
    u = (t - start) / dur
    return 320, _lerp(STAND_HIP_Y, STAND_HIP_Y + 40, u), _lerp(0, 12, u), _lerp(0, 1, u)


def fast_sit(t):
    """Dropping hard onto a chair: high descent velocity but trunk stays upright."""
    return sit_down(t, dur=0.35)


def lie_down_slowly(t, start=3.0):
    """Sit on the floor/sofa (1.5 s), then lie back (2 s): horizontal + low, no impact."""
    if t < start + 1.5:
        u = (t - start) / 1.5
        return 320, _lerp(STAND_HIP_Y, 395, u), _lerp(0, 15, u), _lerp(0, 1, u)
    u = (t - start - 1.5) / 2.0
    return 320, _lerp(395, LYING_HIP_Y, u), _lerp(15, 88, u), _lerp(1, 0, u)


def bend_over(t, start=3.0):
    """Bend to pick something up and straighten again."""
    if t < start + 0.6:
        th = _lerp(0, 75, (t - start) / 0.6)
    elif t < start + 1.6:
        th = 75
    else:
        th = _lerp(75, 0, (t - start - 1.6) / 0.6)
    return 320, STAND_HIP_Y, th, 0.0


def stand_still(t):
    return 320, STAND_HIP_Y, 0.0, 0.0


def walk(t):
    """Walking across the room at ~1.5 body-units/s with a slight bob."""
    return 150 + 90 * t, STAND_HIP_Y + 3 * np.sin(2 * np.pi * 2 * t), 5.0, 0.0


FALLS = {"fall": fall, "fast_fall": fast_fall, "slow_fall": slow_fall}
NON_FALLS = {"sit_down": sit_down, "fast_sit": fast_sit,
             "lie_down_slowly": lie_down_slowly, "bend_over": bend_over}


# ---- feeding the FSM -------------------------------------------------------

class _T:
    """Just enough of a torch tensor for MultiPersonState.step()."""
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return _T(self.a.astype(int))


class _Boxes:
    def __init__(self, xyxy): self.xyxy, self.id = _T(xyxy), _T([1])
    def __len__(self): return len(self.xyxy.a)


class _Kps:
    def __init__(self, xy, conf): self.xy, self.conf = _T(xy[None]), _T(conf[None])


class FakeResult:
    def __init__(self, kpts: np.ndarray, conf: np.ndarray):
        self.boxes = _Boxes([[*kpts.min(0), *kpts.max(0)]])
        self.keypoints = _Kps(kpts, conf)


def run(scenario, fps, state, *, duration=10.0, seed=0, t0=1000.0):
    """Feed `scenario` at `fps` into `state` (a MultiPersonState).

    Returns (first fall-alert time relative to scenario start or None, final state).
    """
    first_alert, _, final = run_states(scenario, fps, state, duration=duration,
                                       seed=seed, t0=t0)
    return first_alert, final


def run_states(scenario, fps, state, *, duration=10.0, seed=0, t0=1000.0):
    """Like run(), also returning the set of FSM states visited."""
    rng = np.random.default_rng(seed)
    first_alert, det, visited = None, None, set()
    for i in range(int(duration * fps)):
        t = i / fps
        kp = skeleton(*scenario(t)) + rng.normal(0, 1.5, (17, 2))
        conf = np.where(rng.random(17) < 0.05, 0.1, 0.9)    # ~5% dropouts
        persons = state.step(FakeResult(kp, conf), 480, 640, t0 + t)
        if persons:
            det = persons[0]["detector"]
            visited.add(det.state)
            if det.fall_alert and first_alert is None:
                first_alert = t
    return first_alert, visited, (det.state if det else None)
