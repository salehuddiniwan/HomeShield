"""OpenCV drawing for fall detection: skeleton, per-person badge, HUD."""

from __future__ import annotations

import time

import cv2

from .fsm import MultiPersonState, State

SKELETON_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                  (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
                  (14, 16), (0, 5), (0, 6)]

STATE_COLOR = {
    State.STANDING: (110, 220, 110),
    State.WALKING: (110, 220, 200),
    State.SITTING: (110, 200, 240),
    State.FALL_DETECTED: (60, 60, 255),
    State.LYING_AFTER_FALL: (50, 130, 255),
    State.LYING_MOTIONLESS: (40, 40, 220),
    State.INACTIVITY: (200, 120, 220),
}


def draw_skeleton(frame, kpts, conf_min):
    for x, y, c in kpts:
        if c >= conf_min:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    for a, b in SKELETON_EDGES:
        if kpts[a, 2] >= conf_min and kpts[b, 2] >= conf_min:
            cv2.line(frame, (int(kpts[a, 0]), int(kpts[a, 1])),
                     (int(kpts[b, 0]), int(kpts[b, 1])), (0, 200, 255), 2)


def draw_person_label(frame, person):
    """Small ID + state badge under each person's bbox."""
    pid = person["id"]
    det = person["detector"]
    bbox = person["bbox"]
    color = STATE_COLOR.get(det.state, (200, 200, 200))
    x1, y1, x2, y2 = map(int, bbox)
    label = f"#{pid} {det.state.value}"
    if det.fall_alert:
        label = f"#{pid} !! FALL !!"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(frame, state: MultiPersonState, primary, fps):
    """Top banner = global alert; side panel = primary (largest) person metrics."""
    h, w = frame.shape[:2]
    n_tracked = len(state.detectors)
    fall_ids = state.fall_alert_ids()
    if fall_ids:
        banner_color = STATE_COLOR[State.FALL_DETECTED]
        ids_str = ", ".join(f"#{i}" for i in fall_ids)
        banner_text = f"!! FALL ALERT !!  Person(s) {ids_str}   ({n_tracked} tracked)"
    elif primary is not None:
        banner_color = STATE_COLOR.get(primary["detector"].state, (200, 200, 200))
        banner_text = (f"{primary['detector'].state.value}   "
                       f"(primary #{primary['id']}, {n_tracked} tracked)")
    else:
        banner_color = (90, 90, 90)
        banner_text = f"No person detected   ({n_tracked} tracked)"

    banner_h = 56
    cv2.rectangle(frame, (0, 0), (w, banner_h), banner_color, -1)
    cv2.putText(frame, banner_text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (20, 20, 20), 2, cv2.LINE_AA)

    if primary is None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, banner_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return

    f = primary["feats"]
    det = primary["detector"]
    now = time.time()
    impact_age = "-"
    if det.last_impact_at is not None:
        impact_age = f"{now - det.last_impact_at:4.1f}s ago"
    lines = [
        f"FPS              : {fps:5.1f}",
        f"trunk angle      : {f.trunk_angle:5.1f} deg",
        f"aspect (w/h)     : {f.aspect_ratio:5.2f}",
        f"height ratio     : {f.height_ratio:5.2f}",
        f"body scale (px)  : {f.body_scale:5.0f}",
        f"centroid v (bu/s): {f.centroid_velocity_bu_s:+6.2f}",
        f"motion (bu/s)    : {f.motion_energy_bu_s:5.2f}",
        "",
        f"last impact      : {impact_age}",
        f"is_horizontal    : {f.is_horizontal}",
        f"is_upright       : {f.is_upright}",
        f"is_low           : {f.is_low}",
        f"is_still         : {f.is_still}",
    ]
    panel_w = 320
    x0, y0 = w - panel_w - 8, banner_h + 8
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 22 * len(lines) + 12),
                  (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 10, y0 + 22 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
