"""Detection back-ends used by the HomeShield pipeline.

  fall   YOLO pose features + per-person fall FSM  (numpy only; drawing in fall.draw)
  fire   YOLO fire/smoke detector wrapper
  face   InsightFace detection + ArcFace embedding + gallery matching

Each detector can also be run on its own:
  python -m homeshield.detectors.fall --source clip.mp4
  python -m homeshield.detectors.fire --source image.jpg
"""
