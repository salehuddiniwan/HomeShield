"""Per-frame skeleton features in body-scale units."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import Config

# COCO-17 keypoint indices (Ultralytics pose models)
KP = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3,
      "right_ear": 4, "left_shoulder": 5, "right_shoulder": 6,
      "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
      "right_wrist": 10, "left_hip": 11, "right_hip": 12,
      "left_knee": 13, "right_knee": 14, "left_ankle": 15,
      "right_ankle": 16}


@dataclass
class FrameFeatures:
    has_person: bool = False
    body_scale: float = 1.0
    trunk_angle: float = 0.0
    aspect_ratio: float = 0.0
    height_ratio: float = 1.0
    centroid_y_norm: float = 0.0
    centroid_velocity_bu_s: float = 0.0
    motion_energy_bu_s: float = 0.0
    is_horizontal: bool = False
    is_upright: bool = False
    is_low: bool = False
    is_still: bool = False


def _midpoint(kpts, a, b, conf_min):
    ka, kb = kpts[a], kpts[b]
    if ka[2] < conf_min or kb[2] < conf_min:
        return None
    return (ka[:2] + kb[:2]) * 0.5


def _kp_bbox(kpts, conf_min):
    valid = kpts[kpts[:, 2] >= conf_min]
    if len(valid) < 4:
        return None
    x1, y1 = float(valid[:, 0].min()), float(valid[:, 1].min())
    x2, y2 = float(valid[:, 0].max()), float(valid[:, 1].max())
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2, y2)


class FeatureExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.smoothed_kpts = None
        self.prev_centroid = None
        self.prev_kpts_motion = None
        self.prev_t = None
        self.height_history = deque()  # (timestamp, h_px)
        self.motion_ref = deque()      # (timestamp, smoothed kpts)

    def reset(self):
        self.smoothed_kpts = None
        self.prev_centroid = None
        self.prev_kpts_motion = None
        self.prev_t = None
        self.height_history.clear()
        self.motion_ref.clear()

    def _smooth(self, kpts):
        a = self.cfg.kp_ema_alpha
        if self.smoothed_kpts is None or self.smoothed_kpts.shape != kpts.shape:
            self.smoothed_kpts = kpts.copy()
            return self.smoothed_kpts
        out = self.smoothed_kpts.copy()
        cur = kpts[:, 2] >= self.cfg.kp_conf_min
        prv = self.smoothed_kpts[:, 2] >= self.cfg.kp_conf_min
        both = cur & prv
        out[both, :2] = a * kpts[both, :2] + (1 - a) * self.smoothed_kpts[both, :2]
        out[both, 2] = kpts[both, 2]
        new_only = cur & ~prv
        out[new_only] = kpts[new_only]
        out[~cur, 2] = 0.0
        self.smoothed_kpts = out
        return out

    def _body_scale(self, kpts):
        sh = _midpoint(kpts, KP["left_shoulder"], KP["right_shoulder"],
                       self.cfg.kp_conf_min)
        hp = _midpoint(kpts, KP["left_hip"], KP["right_hip"],
                       self.cfg.kp_conf_min)
        if sh is not None and hp is not None:
            d = float(np.linalg.norm(sh - hp))
            if d > 1.0:
                return d
        nose = kpts[KP["nose"]]
        if hp is not None and nose[2] >= self.cfg.kp_conf_min:
            d = float(np.linalg.norm(hp - nose[:2])) * 0.6
            if d > 1.0:
                return d
        return None

    def _trunk_angle(self, kpts):
        sh = _midpoint(kpts, KP["left_shoulder"], KP["right_shoulder"],
                       self.cfg.kp_conf_min)
        hp = _midpoint(kpts, KP["left_hip"], KP["right_hip"],
                       self.cfg.kp_conf_min)
        if sh is None or hp is None:
            nose = kpts[KP["nose"]]
            if hp is None or nose[2] < self.cfg.kp_conf_min:
                return None
            sh = nose[:2]
        dx = sh[0] - hp[0]
        dy = hp[1] - sh[1]
        return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-3))))

    def _centroid(self, kpts):
        ok = kpts[:, 2] >= self.cfg.kp_conf_min
        if ok.sum() < 4:
            return None
        return kpts[ok, :2].mean(axis=0)

    def _motion_energy(self, kpts, now, scale):
        """Mean joint speed (body-units / s) relative to the pose about
        `motion_window_s` ago.

        Measuring frame-to-frame instead turns ~1 px of keypoint jitter into
        ~0.6 bu/s at 30 FPS, far above motion_threshold_bu_s, so a person
        lying perfectly still never counted as still (and standing people
        flickered into Walking). Jitter does not accumulate over the window;
        real movement does.
        """
        self.motion_ref.append((now, kpts.copy()))
        cutoff = now - self.cfg.motion_window_s
        while len(self.motion_ref) > 1 and self.motion_ref[1][0] <= cutoff:
            self.motion_ref.popleft()
        ref_t, ref = self.motion_ref[0]
        # Until the baseline spans half the window (first ~0.25 s of a track)
        # the estimate would still be frame-to-frame noise.
        if now - ref_t < 0.5 * self.cfg.motion_window_s or ref.shape != kpts.shape:
            return 0.0
        both = ((kpts[:, 2] >= self.cfg.kp_conf_min) &
                (ref[:, 2] >= self.cfg.kp_conf_min))
        if not both.any():
            return 0.0
        d = kpts[both, :2] - ref[both, :2]
        return float(np.linalg.norm(d, axis=1).mean() / scale / (now - ref_t))

    def _height_ref(self, now, h_px):
        self.height_history.append((now, h_px))
        cutoff = now - self.cfg.height_ref_window_s
        while self.height_history and self.height_history[0][0] < cutoff:
            self.height_history.popleft()
        return max(v for _, v in self.height_history)

    def extract(self, kpts_raw, frame_h, frame_w, now):
        f = FrameFeatures()
        if kpts_raw is None:
            self.reset()
            return f
        kpts = self._smooth(kpts_raw)
        scale = self._body_scale(kpts)
        if scale is None:
            return f
        f.body_scale = scale
        bbox = _kp_bbox(kpts, self.cfg.kp_conf_min)
        if bbox is None:
            return f
        x1, y1, x2, y2 = bbox
        w, h = (x2 - x1), (y2 - y1)
        f.has_person = True
        f.aspect_ratio = w / max(h, 1e-3)
        ref_h = self._height_ref(now, h)
        f.height_ratio = h / max(ref_h, 1e-3)
        ang = self._trunk_angle(kpts)
        f.trunk_angle = ang if ang is not None else (80.0 if f.aspect_ratio > 1.0 else 15.0)
        c = self._centroid(kpts)
        if c is not None:
            f.centroid_y_norm = float(c[1]) / float(frame_h)
            self.prev_centroid = c
        if (self.prev_kpts_motion is not None and self.prev_t is not None
                and self.prev_kpts_motion.shape == kpts.shape):
            both = ((kpts[:, 2] >= self.cfg.kp_conf_min) &
                    (self.prev_kpts_motion[:, 2] >= self.cfg.kp_conf_min))
            if both.sum() >= 4:
                # Centroid descent over joints visible in BOTH frames. Using the
                # centroid of whichever joints are visible makes a single joint
                # flickering below kp_conf_min look like a sudden drop of the
                # whole body, i.e. a fake Stage-A impact.
                dy = kpts[both, 1] - self.prev_kpts_motion[both, 1]
                dt = max(1e-3, now - self.prev_t)
                f.centroid_velocity_bu_s = float(dy.mean()) / scale / dt
        f.motion_energy_bu_s = self._motion_energy(kpts, now, scale)
        self.prev_kpts_motion = kpts.copy()
        self.prev_t = now
        f.is_horizontal = (f.trunk_angle > self.cfg.horizontal_angle_min
                           or f.aspect_ratio > self.cfg.horizontal_aspect)
        f.is_upright = f.trunk_angle < self.cfg.upright_angle_max
        f.is_low = f.height_ratio < self.cfg.height_collapse_ratio
        f.is_still = f.motion_energy_bu_s < self.cfg.motion_threshold_bu_s
        return f
