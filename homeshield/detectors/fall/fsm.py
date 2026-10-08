"""Two-stage fall detector + 7-state FSM, one instance per tracked person."""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np

from .config import Config
from .features import FeatureExtractor


class State(Enum):
    STANDING = "q0 Standing"
    WALKING = "q1 Walking"
    SITTING = "q2 Sitting"
    FALL_DETECTED = "q3 Fall_Detected"
    LYING_AFTER_FALL = "q4 Lying_After_Fall"
    LYING_MOTIONLESS = "q5 Lying_Motionless"
    INACTIVITY = "q6 Inactivity"


# ---- Two-stage detector + FSM --------------------------------------------
class FallDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        max_buf_s = max(cfg.impact_recent_s, cfg.height_ref_window_s,
                        cfg.short_window_s, cfg.sustain_s) + 1.0
        self.history = deque(maxlen=int(cfg.fps_assumed * max_buf_s) + 5)
        self.state = State.STANDING
        self.entered_at = time.time()
        self.last_motion_at = time.time()
        self.last_impact_at = None
        self.fall_alert = False

    def _peak_velocity_in(self, now, window_s):
        cutoff = now - window_s
        peak = 0.0
        for t, f in self.history:
            if t < cutoff or not f.has_person:
                continue
            if f.centroid_velocity_bu_s > peak:
                peak = f.centroid_velocity_bu_s
        return peak

    def _stage_b_sustained(self, now):
        cutoff = now - self.cfg.sustain_s
        rel = [(t, f) for t, f in self.history if t >= cutoff and f.has_person]
        if len(rel) < 3:
            return False
        return all((f.is_horizontal and f.is_low) for _, f in rel)

    def _ratio_in(self, now, window_s, attr):
        cutoff = now - window_s
        rel = [f for t, f in self.history if t >= cutoff and f.has_person]
        if not rel:
            return 0.0
        return sum(1 for f in rel if getattr(f, attr)) / len(rel)

    def _enter(self, new):
        if new != self.state:
            print(f"[FSM] {self.state.value}  ->  {new.value}")
            self.state = new
            self.entered_at = time.time()
            if new == State.FALL_DETECTED:
                self.fall_alert = True
            elif new == State.STANDING:
                self.fall_alert = False

    def step(self, f, now):
        self.history.append((now, f))
        if not f.has_person:
            return
        if not f.is_still:
            self.last_motion_at = now
        cfg = self.cfg
        peak_v = self._peak_velocity_in(now, cfg.short_window_s)
        if peak_v > cfg.impact_velocity_bu_s:
            self.last_impact_at = now
        impact_recent = (self.last_impact_at is not None
                         and (now - self.last_impact_at) <= cfg.impact_recent_s)
        stage_b = self._stage_b_sustained(now)
        s = self.state

        if s in (State.STANDING, State.WALKING, State.SITTING):
            if impact_recent and stage_b:
                self._enter(State.FALL_DETECTED)
                return
            if s == State.STANDING:
                if (f.trunk_angle < cfg.upright_angle_max
                        and f.aspect_ratio < cfg.standing_aspect_max
                        and f.motion_energy_bu_s > cfg.walking_motion_min):
                    self._enter(State.WALKING)
                elif (f.trunk_angle < cfg.horizontal_angle_min
                      and f.aspect_ratio > cfg.sitting_aspect_min
                      and not stage_b):
                    self._enter(State.SITTING)
            elif s == State.WALKING:
                if (f.aspect_ratio > cfg.sitting_aspect_min
                        and f.motion_energy_bu_s < cfg.walking_motion_min
                        and not stage_b):
                    self._enter(State.SITTING)
                elif (f.aspect_ratio < cfg.standing_aspect_max
                      and f.motion_energy_bu_s < cfg.walking_motion_min):
                    self._enter(State.STANDING)
            elif s == State.SITTING:
                if (f.trunk_angle < cfg.upright_angle_max
                        and f.aspect_ratio < cfg.standing_aspect_max):
                    self._enter(State.STANDING)
                elif ((now - self.last_motion_at) > cfg.inactivity_s
                      and self._ratio_in(now, cfg.inactivity_s, "is_still") > 0.7):
                    self._enter(State.INACTIVITY)
            return

        if s == State.FALL_DETECTED:
            time_in = now - self.entered_at
            if stage_b and time_in <= 5.0:
                self._enter(State.LYING_AFTER_FALL)
            elif stage_b and time_in > 5.0:
                self._enter(State.LYING_MOTIONLESS)
            elif (self._ratio_in(now, 1.0, "is_upright") > 0.6
                  and not f.is_horizontal):
                self._enter(State.STANDING)
            return

        if s == State.LYING_AFTER_FALL:
            time_in = now - self.entered_at
            if (time_in > cfg.lying_motionless_s
                    and self._ratio_in(now, 2.0, "is_still") > 0.7):
                self._enter(State.LYING_MOTIONLESS)
            elif (self._ratio_in(now, 1.0, "is_upright") > 0.6
                  and not f.is_horizontal):
                self._enter(State.STANDING)
            return

        if s == State.LYING_MOTIONLESS:
            if (self._ratio_in(now, 1.0, "is_upright") > 0.6
                    and not f.is_horizontal):
                self._enter(State.STANDING)
            return

        if s == State.INACTIVITY:
            if impact_recent and stage_b:
                self._enter(State.FALL_DETECTED)
            elif (self._ratio_in(now, 1.0, "is_upright") > 0.6
                  and not f.is_horizontal):
                self._enter(State.STANDING)


# ---- Multi-person state manager ------------------------------------------
class MultiPersonState:
    """
    Holds one FeatureExtractor + one FallDetector per tracked person ID.
    Cleans up IDs that haven't been seen recently.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extractors: dict = {}      # id -> FeatureExtractor
        self.detectors: dict = {}       # id -> FallDetector
        self.last_seen: dict = {}       # id -> timestamp
        self.last_kpts: dict = {}       # id -> kpts (for drawing)
        self.last_bbox: dict = {}       # id -> bbox (for drawing)
        self.next_anon_id = -1          # used when tracker doesn't return ids

    def _get_or_create(self, pid: int):
        if pid not in self.extractors:
            self.extractors[pid] = FeatureExtractor(self.cfg)
            self.detectors[pid] = FallDetector(self.cfg)
        return self.extractors[pid], self.detectors[pid]

    def _cleanup(self, now: float) -> None:
        cutoff = now - self.cfg.id_cleanup_after_s
        stale = [pid for pid, t in self.last_seen.items() if t < cutoff]
        for pid in stale:
            self.extractors.pop(pid, None)
            self.detectors.pop(pid, None)
            self.last_seen.pop(pid, None)
            self.last_kpts.pop(pid, None)
            self.last_bbox.pop(pid, None)

    def step(self, result, frame_h: int, frame_w: int, now: float) -> list:
        """
        Process a YOLO tracking result. Returns list of dicts:
        [{'id': int, 'kpts': ndarray, 'bbox': ndarray,
          'feats': FrameFeatures, 'detector': FallDetector}, ...]
        """
        out = []
        if (result.boxes is None or len(result.boxes) == 0
                or result.keypoints is None):
            self._cleanup(now)
            return out

        boxes = result.boxes.xyxy.cpu().numpy()
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_c = (result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(kp_xy.shape[:2], dtype=np.float32))

        # Track IDs (None when track() wasn't used or tracker has no IDs yet)
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy().tolist()
        else:
            ids = [self._fresh_anon_id() for _ in range(len(boxes))]

        for i, pid in enumerate(ids):
            pid = int(pid)
            kpts = np.concatenate([kp_xy[i], kp_c[i][:, None]], axis=1)
            bbox = boxes[i]
            extractor, detector = self._get_or_create(pid)
            feats = extractor.extract(kpts, frame_h, frame_w, now)
            detector.step(feats, now)
            self.last_seen[pid] = now
            self.last_kpts[pid] = kpts
            self.last_bbox[pid] = bbox
            out.append({
                "id": pid, "kpts": kpts, "bbox": bbox,
                "feats": feats, "detector": detector,
            })

        self._cleanup(now)
        return out

    def _fresh_anon_id(self) -> int:
        self.next_anon_id -= 1
        return self.next_anon_id

    # --- queries used by drawing -----------------------------------------
    def any_fall_alert(self) -> bool:
        return any(d.fall_alert for d in self.detectors.values())

    def fall_alert_ids(self) -> list:
        return [pid for pid, d in self.detectors.items() if d.fall_alert]

    def primary_person(self, persons: list) -> Optional[dict]:
        """Largest-bbox person (used for the side metrics panel)."""
        if not persons:
            return None
        def area(p):
            x1, y1, x2, y2 = p["bbox"]
            return float((x2 - x1) * (y2 - y1))
        return max(persons, key=area)
