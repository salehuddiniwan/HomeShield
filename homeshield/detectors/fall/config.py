"""All fall-detection thresholds in one dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Bare file names are looked up in the CWD, then in Fall_Detection/weights/.
    model_path: str = "yolo26x-pose.pt"
    fallback_model: str = "yolo11n-pose.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.35
    kp_conf_min: float = 0.30

    kp_ema_alpha: float = 0.6
    fps_assumed: float = 30.0
    short_window_s: float = 0.5         # peak-velocity window
    impact_recent_s: float = 2.0        # how far back the peak still counts
    sustain_s: float = 0.5              # Stage-B must hold this long
    lying_motionless_s: float = 5.0
    inactivity_s: float = 5.0
    height_ref_window_s: float = 3.0

    # Spatial thresholds
    upright_angle_max: float = 30.0
    horizontal_angle_min: float = 60.0
    walking_motion_min: float = 0.05    # body-units / s
    sitting_aspect_min: float = 1.0
    standing_aspect_max: float = 0.7
    horizontal_aspect: float = 1.3
    height_collapse_ratio: float = 0.55

    # Temporal thresholds (BODY-UNITS / SECOND)
    impact_velocity_bu_s: float = 1.6
    motion_threshold_bu_s: float = 0.10

    # Multi-person tracking
    tracker: str = "bytetrack.yaml"        # "bytetrack.yaml" | "botsort.yaml"
    id_cleanup_after_s: float = 2.0        # remove IDs unseen this long
    max_persons_drawn: int = 16

    show: bool = True
    save_path: Optional[str] = None
    draw_skeleton: bool = True
    draw_hud: bool = True
