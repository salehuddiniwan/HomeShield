"""Filesystem locations shared by the dashboard and the detector CLIs.

Paths are resolved relative to the source checkout, so HomeShield must be
run from the repo (or installed with `pip install -e .`).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSE_WEIGHTS_DIR = PROJECT_ROOT / "Fall_Detection" / "weights"
FIRE_WEIGHTS_DIR = PROJECT_ROOT / "Fire_Detection"
