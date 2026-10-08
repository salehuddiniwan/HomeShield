"""
Fire / smoke detection using local YOLO weights.

Library use (from the pipeline):
    fires = predict(model, frame, conf=0.35, imgsz=640, device=0)
    # -> [{"bbox": (x1, y1, x2, y2), "conf": float, "cls_name": "fire"}, ...]

Standalone inference on images, videos, or a webcam:
    python -m homeshield.detectors.fire --source path/to/image.jpg
    python -m homeshield.detectors.fire --source path/to/folder
    python -m homeshield.detectors.fire --source path/to/video.mp4
    python -m homeshield.detectors.fire --source 0 --show
    python -m homeshield.detectors.fire --source image.jpg --conf 0.4 --out runs/fire
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import FIRE_WEIGHTS_DIR

DEFAULT_WEIGHTS = FIRE_WEIGHTS_DIR / "best.pt"


def load_model(weights_path: str):
    """Load the YOLO model from a local weights file."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(
            "Missing dependency 'ultralytics'. Install it with:\n"
            "    pip install ultralytics\n"
            f"(original error: {e})"
        )
    return YOLO(weights_path)


def parse_result(r0) -> list[dict[str, Any]]:
    """Flatten one Ultralytics detect result into homeshield fire dicts."""
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    names = getattr(r0, "names", None) or {}
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    clss = r0.boxes.cls.cpu().numpy().astype(int)
    if xyxy.ndim != 2 or xyxy.shape[1] < 4:
        xyxy = np.zeros((0, 4), dtype=np.float32)
        confs = np.zeros((0,), dtype=np.float32)
        clss = np.zeros((0,), dtype=int)
    fires: list[dict[str, Any]] = []
    for row, c, k in zip(xyxy, confs, clss):
        if len(row) < 4:
            continue
        fires.append({
            "bbox": (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
            "conf": float(c),
            "cls_name": str(names.get(int(k), str(k))).lower(),
        })
    return fires


def predict(model, frame, *, conf: float, imgsz: int, device,
            half: bool = False) -> list[dict[str, Any]]:
    """Run the fire model on one BGR frame. Not thread-safe: callers serialize."""
    fr = model.predict(frame, conf=conf, imgsz=imgsz, device=device,
                       half=half, verbose=False)
    return parse_result(fr[0]) if fr else []


class FireConfirmer:
    """Temporal confirmation: a class counts as present only once it has been
    detected in at least `k` of the last `n` fire inference runs.

    Single-frame detections on lamps, sunsets, orange clothing or screens are
    the main source of fire false alarms; real fire and smoke persist.
    """

    def __init__(self) -> None:
        self._hist: dict[str, deque] = {}

    def update(self, present: set[str], k: int, n: int) -> set[str]:
        n = max(1, int(n))
        k = min(max(1, int(k)), n)
        for cls in set(self._hist) | set(present):
            h = self._hist.get(cls)
            if h is None or h.maxlen != n:
                h = self._hist[cls] = deque(h or (), maxlen=n)
            h.append(cls in present)
            if not any(h):
                del self._hist[cls]
        return {cls for cls, h in self._hist.items() if sum(h) >= k}

    def reset(self) -> None:
        self._hist.clear()


# ---- CLI ------------------------------------------------------------------

def parse_source(src: str):
    """Treat purely numeric source as a webcam device index."""
    if src.isdigit():
        return int(src)
    return src


def main():
    parser = argparse.ArgumentParser(
        prog="python -m homeshield.detectors.fire",
        description="YOLO fire detection inference")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to the local YOLO weights file (default: Fire_Detection/best.pt)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, folder, video path, or webcam index (e.g. 0)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--out",
        default="runs/fire",
        help="Folder to save annotated outputs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window (good for webcam)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device, e.g. 'cpu', '0', '0,1'. Default: auto",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated outputs to disk",
    )
    args = parser.parse_args()

    # Verify the local weights file exists
    if not Path(args.weights).is_file():
        sys.exit(f"[error] Weights file '{args.weights}' not found. Please provide a valid path using --weights.")

    print(f"[info] Loading local model weights from: {args.weights}")
    model = load_model(args.weights)

    source = parse_source(args.source)
    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=not args.no_save,
        show=args.show,
        project=str(out_dir.parent) if out_dir.parent != Path("") else "runs",
        name=out_dir.name,
        exist_ok=True,
        stream=False,
    )

    # Quick textual summary
    total_dets = 0
    for r in results:
        if r.boxes is not None:
            total_dets += len(r.boxes)
    print(f"[done] {len(results)} frame(s) processed, {total_dets} detection(s) total.")
    if not args.no_save:
        print(f"[done] Annotated output saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
