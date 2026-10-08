"""Standalone fall-detection runner (webcam or video file).

  python -m homeshield.detectors.fall
  python -m homeshield.detectors.fall --source clip.mp4
  python -m homeshield.detectors.fall --source clip.mp4 --save out.mp4 --no-show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from ...paths import POSE_WEIGHTS_DIR
from .config import Config
from .draw import STATE_COLOR, draw_hud, draw_person_label, draw_skeleton
from .fsm import MultiPersonState


# ---- Model loading -------------------------------------------------------
def _resolve_model_path(cfg):
    for candidate in (Path(cfg.model_path),
                      POSE_WEIGHTS_DIR / Path(cfg.model_path).name):
        if candidate.is_file():
            return str(candidate)
    print(f"[model] '{cfg.model_path}' not found, using '{cfg.fallback_model}'.")
    return cfg.fallback_model


def load_model(cfg):
    model = YOLO(_resolve_model_path(cfg))
    if cfg.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    try:
        model.to(0 if device == "0" else device)
    except Exception as e:
        print(f"[model] .to({device}) failed ({e}); using default device.")
    return model, device


# ---- Main loop -----------------------------------------------------------
def run(cfg, source):
    model, device = load_model(cfg)
    print(f"[device] running on: {device} (cuda: {torch.cuda.is_available()})")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_assumed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cfg.fps_assumed = float(src_fps if src_fps > 0 else cfg.fps_assumed)
    writer = None
    if cfg.save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.save_path, fourcc, src_fps, (width, height))
        print(f"[save] writing -> {cfg.save_path}")

    state = MultiPersonState(cfg)
    smoothed_fps = float(src_fps)
    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # model.track() runs detection + persistent ID assignment via ByteTrack
        results = model.track(frame, imgsz=cfg.imgsz, conf=cfg.conf,
                              persist=True, tracker=cfg.tracker,
                              verbose=False,
                              device=0 if device == "0" else device)
        result = results[0]
        now = time.time()
        persons = state.step(result, frame.shape[0], frame.shape[1], now)

        # Per-person drawing (capped so a crowded scene doesn't spam the screen)
        for person in persons[:cfg.max_persons_drawn]:
            if cfg.draw_skeleton and person["kpts"] is not None:
                draw_skeleton(frame, person["kpts"], cfg.kp_conf_min)
            x1, y1, x2, y2 = map(int, person["bbox"])
            color = STATE_COLOR.get(person["detector"].state, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_person_label(frame, person)

        dt = max(1e-6, now - last_t)
        last_t = now
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / dt)

        if cfg.draw_hud:
            primary = state.primary_person(persons)
            draw_hud(frame, state, primary, smoothed_fps)

        if writer is not None:
            writer.write(frame)
        if cfg.show:
            cv2.imshow("YOLO26 Fall Detection - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m homeshield.detectors.fall",
        description="YOLO26 multi-person fall detection (research-tuned)")
    p.add_argument("--source", default="0",
                   help="Webcam index (0/1/...) or video file path")
    p.add_argument("--model", default=Config.model_path,
                   help="Weights path, or a file name inside Fall_Detection/weights/")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "0"])
    p.add_argument("--imgsz", type=int, default=Config.imgsz)
    p.add_argument("--conf", type=float, default=Config.conf)
    p.add_argument("--tracker", default=Config.tracker,
                   choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--impact-vel", type=float,
                   default=Config.impact_velocity_bu_s,
                   help="Stage-A peak descent threshold (body-units/sec)")
    p.add_argument("--horizontal-deg", type=float,
                   default=Config.horizontal_angle_min,
                   help="Trunk angle (deg) considered horizontal")
    p.add_argument("--collapse-ratio", type=float,
                   default=Config.height_collapse_ratio,
                   help="height_ratio below this = collapsed")
    p.add_argument("--save", default=None, help="Path to write annotated mp4")
    p.add_argument("--no-show", action="store_true",
                   help="Disable display window")
    args = p.parse_args()
    cfg = Config(model_path=args.model, device=args.device, imgsz=args.imgsz,
                 conf=args.conf, tracker=args.tracker,
                 impact_velocity_bu_s=args.impact_vel,
                 horizontal_angle_min=args.horizontal_deg,
                 height_collapse_ratio=args.collapse_ratio,
                 show=not args.no_show, save_path=args.save)
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    return cfg, src


if __name__ == "__main__":
    cfg, src = parse_args()
    run(cfg, src)
