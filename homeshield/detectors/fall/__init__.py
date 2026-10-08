"""
Fall Detection with YOLO26 Pose Estimation  (research-tuned rewrite)
====================================================================

Design choices, with the papers they came from:

- Body-scale normalization. Distances/velocities are expressed in
  *body-units* (1 unit = ||shoulder_mid - hip_mid||), so metrics are
  independent of camera distance. (Nunez-Marcos et al., Pattern
  Recognition Letters 2022; PMC9185346 on 2D skeleton normalization.)

- Trunk-angle from shoulder->hip vector vs. vertical. Falls are
  characterised by trunk angle > ~45-60 deg. We fall back to the
  head->hip vector when shoulders are occluded. (MDPI Symmetry 2020,
  "Fall Detection Based on Key Points of Human-Skeleton Using
  OpenPose".)

- Two-stage decision. (A) impact = peak centroid descent velocity
  exceeded threshold. (B) lying = sustained horizontal + low. Both
  must fire within a short window, which rejects controlled
  sit-downs. (PIFR PLOS ONE 2024; arXiv 2401.01587.)

- Height-ratio against the last-3s maximum, not a decaying running
  max. current_h / recent_max_h drops below ~0.55 during a fall.
  (Sensors 2025 Enhanced HRNet+YOLO.)

- Confidence-weighted keypoints + EMA smoothing.

Modules:
  config    Config dataclass (all thresholds)
  features  per-frame skeleton features            (numpy only)
  fsm       two-stage detector + per-person FSM     (numpy only)
  draw      skeleton / label / HUD drawing          (cv2)
  __main__  standalone webcam / video runner        (torch + ultralytics)

Run:
  python -m homeshield.detectors.fall
  python -m homeshield.detectors.fall --source clip.mp4
  python -m homeshield.detectors.fall --source clip.mp4 --save out.mp4 --no-show
"""

from .config import Config
from .features import KP, FeatureExtractor, FrameFeatures
from .fsm import FallDetector, MultiPersonState, State

__all__ = [
    "Config", "KP", "FeatureExtractor", "FrameFeatures",
    "FallDetector", "MultiPersonState", "State",
]
