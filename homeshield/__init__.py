"""HomeShield: Centralized Dashboard for Real-Time CCTV Monitoring and Anomaly Detection."""

import os

# Ultralytics "auto-installs" missing packages by running whatever `pip` is on
# PATH, which can belong to a different Python than the one running HomeShield
# (it installed lapx + NumPy 2 into the system Python from a venv). Fail with a
# clear ImportError instead; set YOLO_AUTOINSTALL=true to opt back in.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
