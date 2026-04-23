"""
HomeShield Detector — GPU-accelerated YOLOv8-Pose (detection + keypoints in one pass).

Replaces MediaPipe entirely.  YOLOv8-pose outputs 17 COCO keypoints on the GPU,
so there is no CPU pose step and no thread pool needed for pose estimation.

COCO-17 keypoint indices used throughout:
  0  nose          5  l_shoulder    6  r_shoulder
  7  l_elbow       8  r_elbow       9  l_wrist    10  r_wrist
  11 l_hip        12  r_hip        13  l_knee     14  r_knee
  15 l_ankle      16  r_ankle
"""

import cv2
import numpy as np
import time
import math
import threading
from collections import deque
from config import Config

# ── Optional torch (for device selection only) ────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed")

OPENCV_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False


# ── Device resolution ─────────────────────────────────────────────────────────

def _resolve_device() -> str:
    cfg = Config.GPU_DEVICE.strip().lower()
    if cfg != "auto":
        return cfg
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[GPU] CUDA: {props.name}  VRAM: {props.total_memory/1024**3:.1f} GB")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Using Apple MPS")
            return "mps"
    print("[GPU] No GPU — falling back to CPU")
    return "cpu"


DEVICE   = _resolve_device()
USE_FP16 = Config.USE_FP16 and DEVICE not in ("cpu",)


# ── Lightweight landmark wrapper ──────────────────────────────────────────────

class _KP:
    """Wraps a single COCO keypoint (x, y, conf) — mimics the .x/.y/.visibility API."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, conf: float):
        self.x          = x           # normalised [0, 1]
        self.y          = y           # normalised [0, 1]
        self.visibility = conf        # confidence [0, 1]


def _build_landmarks(kp_tensor, frame_w: int, frame_h: int):
    """
    Convert a (17, 3) keypoint tensor (pixel x, pixel y, conf)
    to a list of 17 _KP objects with normalised coordinates.
    """
    lm = []
    for row in kp_tensor:
        px, py, conf = float(row[0]), float(row[1]), float(row[2])
        lm.append(_KP(px / frame_w, py / frame_h, conf))
    return lm


# ── COCO-17 keypoint indices ──────────────────────────────────────────────────

class KP:
    NOSE        = 0
    L_EYE, R_EYE           = 1, 2
    L_EAR, R_EAR           = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW,    R_ELBOW    = 7, 8
    L_WRIST,    R_WRIST    = 9, 10
    L_HIP,      R_HIP      = 11, 12
    L_KNEE,     R_KNEE     = 13, 14
    L_ANKLE,    R_ANKLE    = 15, 16


# ── Person tracker ────────────────────────────────────────────────────────────

class PersonTracker:

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict = {}
        self.disappeared: dict = {}
        self.histories: dict = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        pid = self.next_id
        self.objects[pid] = {"centroid": centroid, "bbox": bbox}
        self.disappeared[pid] = 0
        self.histories[pid] = deque(maxlen=60)
        self.histories[pid].append({"centroid": centroid, "bbox": bbox, "time": time.time()})
        self.next_id += 1

    def deregister(self, pid):
        del self.objects[pid]
        del self.disappeared[pid]
        del self.histories[pid]

    def update(self, detections):
        if not detections:
            for pid in list(self.disappeared):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    self.deregister(pid)
            return self.objects

        input_centroids, input_bboxes = [], []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            input_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
            input_bboxes.append(det["bbox"])

        if not self.objects:
            for c, b in zip(input_centroids, input_bboxes):
                self.register(c, b)
        else:
            obj_ids = list(self.objects)
            obj_centroids = [self.objects[p]["centroid"] for p in obj_ids]

            dists = np.array([[math.dist(oc, ic) for ic in input_centroids]
                               for oc in obj_centroids])
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]
            used_r, used_c = set(), set()

            for r, c in zip(rows, cols):
                if r in used_r or c in used_c or dists[r, c] > 150:
                    continue
                pid = obj_ids[r]
                self.objects[pid] = {"centroid": input_centroids[c], "bbox": input_bboxes[c]}
                self.disappeared[pid] = 0
                self.histories[pid].append({
                    "centroid": input_centroids[c], "bbox": input_bboxes[c], "time": time.time()
                })
                used_r.add(r); used_c.add(c)

            for r in set(range(len(obj_ids))) - used_r:
                pid = obj_ids[r]
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    self.deregister(pid)
            for c in set(range(len(input_centroids))) - used_c:
                self.register(input_centroids[c], input_bboxes[c])

        return self.objects


# ── Fall detector ─────────────────────────────────────────────────────────────

class FallDetector:
    """Rule-based fall detection using COCO-17 keypoints."""

    VIS_THRESH = 0.4

    def __init__(self):
        self.pose_history:     dict = {}
        self.fall_states:      dict = {}
        self.inactivity_timers:dict = {}

    def _mid(self, lm, a, b):
        return ((lm[a].x + lm[b].x) / 2, (lm[a].y + lm[b].y) / 2)

    def _vis(self, lm, *idx):
        return all(lm[i].visibility >= self.VIS_THRESH for i in idx)

    def _body_angle(self, lm):
        """Angle of torso from vertical (0=upright, 90=horizontal)."""
        if not self._vis(lm, KP.L_SHOULDER, KP.R_SHOULDER, KP.L_HIP, KP.R_HIP):
            return 0.0
        ms = self._mid(lm, KP.L_SHOULDER, KP.R_SHOULDER)
        mh = self._mid(lm, KP.L_HIP, KP.R_HIP)
        return abs(math.degrees(math.atan2(ms[0]-mh[0], -(ms[1]-mh[1]))))

    def _body_height_ratio(self, lm):
        """Vertical-to-horizontal span ratio (<1.2 = lying flat)."""
        idx = [KP.NOSE, KP.L_SHOULDER, KP.R_SHOULDER,
               KP.L_HIP, KP.R_HIP, KP.L_ANKLE, KP.R_ANKLE]
        vis_idx = [i for i in idx if lm[i].visibility >= self.VIS_THRESH]
        if len(vis_idx) < 3:
            return 2.0
        xs = [lm[i].x for i in vis_idx]
        ys = [lm[i].y for i in vis_idx]
        return (max(ys)-min(ys)) / (max(xs)-min(xs) + 1e-6)

    def _hip_velocity(self, pid):
        hist = self.pose_history.get(pid)
        if not hist or len(hist) < 3:
            return 0.0
        h = list(hist)
        dt = h[-1]["time"] - h[-3]["time"]
        return (h[-1]["hip_y"] - h[-3]["hip_y"]) / dt if dt > 0.01 else 0.0

    def analyze(self, pid, lm, frame_height):
        now = time.time()
        if pid not in self.pose_history:
            self.pose_history[pid]        = deque(maxlen=30)
            self.fall_states[pid]         = {"fallen": False, "fall_time": 0}
            self.inactivity_timers[pid]   = now

        angle = self._body_angle(lm)
        ratio = self._body_height_ratio(lm)
        hip_y = self._mid(lm, KP.L_HIP, KP.R_HIP)[1] \
                if self._vis(lm, KP.L_HIP, KP.R_HIP) else 0.5

        self.pose_history[pid].append(
            {"angle": angle, "ratio": ratio, "hip_y": hip_y, "time": now})
        hip_vel = self._hip_velocity(pid)

        hist = list(self.pose_history[pid])
        if len(hist) >= 2:
            prev = hist[-2]
            if abs(hip_y-prev["hip_y"]) + abs(angle-prev["angle"]) > 0.015:
                self.inactivity_timers[pid] = now

        is_h   = angle > 50
        is_w   = ratio < 1.3
        rapid  = hip_vel > 0.5
        state  = self.fall_states[pid]

        if not state["fallen"]:
            if is_h and is_w and rapid:
                state.update({"fallen": True, "fall_time": now})
                return "fall_detected", min(0.99, 0.6+(angle-50)/80+hip_vel*0.15)
            if is_h and is_w and len(hist) > 5:
                if any(h["angle"] < 45 for h in hist[-6:-1]):
                    state.update({"fallen": True, "fall_time": now})
                    return "fall_detected", min(0.92, 0.5+(angle-50)/100)
            if angle > 70 and ratio < 0.9:
                state.update({"fallen": True, "fall_time": now})
                return "fall_detected", min(0.88, 0.55+(angle-70)/100)

        if state["fallen"] and is_h:
            elapsed = now - state["fall_time"]
            return ("lying_motionless", min(0.95, 0.7+elapsed/100)) \
                   if elapsed > 5 else ("lying_after_fall", 0.85)

        if state["fallen"] and not is_h:
            state["fallen"] = False

        inactive = now - self.inactivity_timers.get(pid, now)
        if inactive > Config.INACTIVITY_SECONDS:
            return "inactivity", min(0.95, 0.6+inactive/1000)

        if angle < 20 and ratio > 2.0: return "standing", 0.9
        if angle < 35 and ratio > 1.0: return "sitting",  0.85
        if angle < 30:                  return "walking",   0.8
        return "unknown", 0.5


# ── Face / age estimator ──────────────────────────────────────────────────────

class FaceAgeEstimator:
    """
    Estimates child / adult / elderly using COCO-17 pose keypoints.

    Signals (in priority order):
      1. Head-to-body ratio from pose keypoints  (children ≈ 1/4, adults ≈ 1/7)
      2. Haar cascade face size vs person height
      3. Bounding-box height fallback
    Results are cached per person_id for CACHE_TTL seconds.
    """

    CACHE_TTL     = 12.0
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    VIS           = 0.4

    def __init__(self):
        self._cache: dict = {}
        self._lock  = threading.Lock()
        self._cascade = cv2.CascadeClassifier(self._CASCADE_PATH)
        if self._cascade.empty():
            self._cascade = None
        print("[INFO] FaceAgeEstimator ready (YOLOv8-pose keypoints + Haar cascade)")

    def classify(self, pid, frame, bbox, landmarks=None):
        now = time.time()
        with self._lock:
            e = self._cache.get(pid)
            if e and (now - e["time"]) < self.CACHE_TTL:
                return e["category"], e["confidence"]
        cat, conf = self._estimate(frame, bbox, landmarks)
        with self._lock:
            self._cache[pid] = {"category": cat, "confidence": conf, "time": now}
        return cat, conf

    def evict(self, pid):
        with self._lock:
            self._cache.pop(pid, None)

    def _estimate(self, frame, bbox, lm):
        fh = frame.shape[0]

        if lm is not None:
            r = self._pose_estimate(lm)
            if r and r[1] >= 0.68:
                return r

        face_r = self._haar_estimate(frame, bbox)
        if face_r:
            return face_r

        if lm is not None:
            r = self._pose_estimate(lm)
            if r:
                return r

        return self._bbox_fallback(bbox, fh)

    def _pose_estimate(self, lm):
        try:
            if not all(lm[i].visibility > self.VIS
                       for i in [KP.NOSE, KP.L_SHOULDER, KP.R_SHOULDER]):
                return None

            nose_y   = lm[KP.NOSE].y
            mid_sh_y = (lm[KP.L_SHOULDER].y + lm[KP.R_SHOULDER].y) / 2
            head_h   = abs(mid_sh_y - nose_y)

            if all(lm[i].visibility > self.VIS for i in [KP.L_ANKLE, KP.R_ANKLE]):
                body_h = abs((lm[KP.L_ANKLE].y + lm[KP.R_ANKLE].y) / 2 - nose_y)
            elif all(lm[i].visibility > self.VIS for i in [KP.L_HIP, KP.R_HIP]):
                body_h = abs((lm[KP.L_HIP].y + lm[KP.R_HIP].y) / 2 - nose_y) * 2.0
            else:
                return None

            if body_h < 0.05:
                return None

            ratio = head_h / body_h

            if ratio > 0.22:
                return "child", min(0.93, 0.72 + (ratio - 0.22) * 4)
            if ratio > 0.17:
                return "child", 0.68

            # Elderly: stooped posture → narrow shoulder/torso ratio
            if all(lm[i].visibility > self.VIS for i in [KP.L_HIP, KP.R_HIP]):
                sh_w   = abs(lm[KP.L_SHOULDER].x - lm[KP.R_SHOULDER].x)
                torso  = abs((lm[KP.L_HIP].y+lm[KP.R_HIP].y)/2 - mid_sh_y)
                if torso > 0 and sh_w / torso < 0.70:
                    return "elderly", 0.60

            return "adult", 0.72
        except Exception:
            return None

    def _haar_estimate(self, frame, bbox):
        if self._cascade is None:
            return None
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        crop = frame[max(0,y1-15):min(fh,y2+15), max(0,x1-15):min(fw,x2+15)]
        if crop.size == 0:
            return None
        person_h = y2 - y1
        gray  = cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(20, person_h//8), max(20, person_h//8)))
        if not len(faces):
            return None
        _, _, _, fh_face = max(faces, key=lambda f: f[2]*f[3])
        r = fh_face / (person_h + 1e-6)
        if r > 0.30: return "child", min(0.88, 0.68 + (r-0.30)*2)
        if r > 0.24: return "child", 0.62
        return None

    @staticmethod
    def _bbox_fallback(bbox, frame_height):
        _, y1, _, y2 = bbox
        r = (y2-y1) / frame_height
        if r < 0.35: return "child",   0.45
        if r > 0.55: return "adult",   0.45
        return "elderly", 0.40


# ── Skeleton drawing ──────────────────────────────────────────────────────────

# COCO-17 bone connections for skeleton overlay
_COCO_CONNECTIONS = [
    (KP.NOSE, KP.L_EAR), (KP.NOSE, KP.R_EAR),
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_ELBOW), (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW), (KP.R_ELBOW, KP.R_WRIST),
    (KP.L_SHOULDER, KP.L_HIP),   (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP,      KP.R_HIP),
    (KP.L_HIP,  KP.L_KNEE),  (KP.L_KNEE,  KP.L_ANKLE),
    (KP.R_HIP,  KP.R_KNEE),  (KP.R_KNEE,  KP.R_ANKLE),
]
_JOINT_IDX = list(range(17))
_VIS_DRAW  = 0.35   # lower threshold for drawing (more lenient than analysis)


def draw_skeleton(frame, lm, color, fw, fh):
    """Draw COCO-17 skeleton on frame. lm = list of _KP objects."""
    def px(i):
        p = lm[i]
        if p.visibility < _VIS_DRAW:
            return None
        return (int(np.clip(p.x * fw, 0, fw-1)),
                int(np.clip(p.y * fh, 0, fh-1)))

    for i, j in _COCO_CONNECTIONS:
        p1, p2 = px(i), px(j)
        if p1 and p2:
            cv2.line(frame, p1, p2, (255, 255, 255), 3, cv2.LINE_AA)

    for idx in _JOINT_IDX:
        pt = px(idx)
        if pt:
            cv2.circle(frame, pt, 6, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, color, -1)


# ── Main detector ─────────────────────────────────────────────────────────────

class Detector:
    """
    Single-model GPU pipeline: YOLOv8-pose outputs person bboxes + 17 COCO
    keypoints in ONE forward pass.  No MediaPipe, no thread pool.
    """

    _COLORS = {
        "elderly": (0, 165, 255),
        "child":   (0, 220, 80),
        "adult":   (255, 180, 0),
    }

    def __init__(self):
        print(f"[GPU] Inference device : {DEVICE}")
        print(f"[GPU] FP16 half-prec   : {USE_FP16}")
        print(f"[GPU] OpenCV CUDA      : {OPENCV_CUDA}")
        print(f"[GPU] Batch size       : {Config.INFERENCE_BATCH_SIZE}")

        self.yolo = None
        if YOLO_AVAILABLE:
            # Use pose model — downloads yolov8n-pose.pt automatically on first run
            pose_model = Config.YOLO_MODEL.replace(".pt", "-pose.pt") \
                         if "-pose" not in Config.YOLO_MODEL else Config.YOLO_MODEL
            try:
                self.yolo = YOLO(pose_model)
                self.yolo.to(DEVICE)
                print(f"[INFO] YOLOv8-pose on {DEVICE} (fp16={USE_FP16}): {pose_model}")
                self._warmup()
            except Exception as e:
                print(f"[WARN] Pose model failed ({e}), trying detection model")
                try:
                    self.yolo = YOLO(Config.YOLO_MODEL)
                    self.yolo.to(DEVICE)
                    print(f"[INFO] YOLOv8 (no pose) on {DEVICE}: {Config.YOLO_MODEL}")
                    self._warmup()
                except Exception as e2:
                    print(f"[WARN] YOLO load failed: {e2}")

        # HOG fallback
        if self.yolo is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[INFO] Using HOG fallback")

        self.fall_detector  = FallDetector()
        self.age_estimator  = FaceAgeEstimator()
        self.tracker        = PersonTracker()

    def _warmup(self):
        dummy = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=np.uint8)
        for _ in range(Config.WARMUP_FRAMES):
            self.yolo(dummy, imgsz=Config.YOLO_IMGSZ, device=DEVICE,
                      half=USE_FP16, verbose=False)
        print(f"[GPU] YOLO warmed up ({Config.WARMUP_FRAMES} frames)")

    # ── Detection + pose ──────────────────────────────────────────────────────

    def _infer_batch(self, frames):
        """
        Returns list of (detections, landmarks_list) per frame.
        detections     = [{"bbox": (x1,y1,x2,y2), "confidence": f}, ...]
        landmarks_list = [list-of-17-_KP or None, ...]  parallel to detections
        """
        if not frames:
            return []

        if self.yolo is not None:
            results = self.yolo(
                frames,
                imgsz=Config.YOLO_IMGSZ,
                conf=Config.YOLO_CONFIDENCE,
                classes=[0],
                device=DEVICE,
                half=USE_FP16,
                verbose=False,
                stream=False,
            )
            out = []
            fh, fw = frames[0].shape[:2]
            for r in results:
                dets, kp_list = [], []
                has_pose = hasattr(r, "keypoints") and r.keypoints is not None \
                           and len(r.keypoints) > 0
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append({"bbox": (x1,y1,x2,y2),
                                 "confidence": float(box.conf[0])})
                    if has_pose and i < len(r.keypoints):
                        kp_data = r.keypoints[i].data
                        if kp_data is not None and len(kp_data) > 0:
                            kp_list.append(_build_landmarks(
                                kp_data[0].cpu().numpy(), fw, fh))
                        else:
                            kp_list.append(None)
                    else:
                        kp_list.append(None)
                out.append((dets, kp_list))
            return out

        # HOG fallback — no keypoints
        out = []
        for frame in frames:
            boxes, weights = self.hog.detectMultiScale(
                frame, winStride=(8,8), padding=(4,4), scale=1.05)
            dets = [{"bbox":(x,y,x+w,y+h), "confidence":float(wt)}
                    for (x,y,w,h), wt in zip(boxes, weights)]
            out.append((dets, [None]*len(dets)))
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame, camera_id=None, zones=None):
        return self.process_frames_batch([(frame, camera_id, zones or [])])[0]

    def process_frames_batch(self, camera_inputs):
        if not camera_inputs:
            return []

        frames     = [ci[0] for ci in camera_inputs]
        cam_ids    = [ci[1] for ci in camera_inputs]
        zones_list = [ci[2] for ci in camera_inputs]

        # Single GPU inference pass: detection + pose for all cameras
        infer_results = self._infer_batch(frames)

        outputs = []
        for frame, camera_id, zones, (detections, kp_list) in \
                zip(frames, cam_ids, zones_list, infer_results):

            fh, fw = frame.shape[:2]
            annotated = frame.copy()
            events    = []

            # Build detection→keypoint map before tracking (index-matched)
            kp_by_det = {i: kp for i, kp in enumerate(kp_list)}

            # Track
            before_ids = set(self.tracker.objects.keys())
            self.tracker.update(detections)
            after_ids  = set(self.tracker.objects.keys())
            for gone in before_ids - after_ids:
                self.age_estimator.evict(gone)

            # Map tracker person_ids → keypoints via centroid matching
            pid_to_kp: dict = {}
            if detections:
                det_centroids = [
                    ((d["bbox"][0]+d["bbox"][2])//2,
                     (d["bbox"][1]+d["bbox"][3])//2)
                    for d in detections
                ]
                for pid, info in self.tracker.objects.items():
                    pc = info["centroid"]
                    best_i = min(range(len(det_centroids)),
                                 key=lambda i: math.dist(pc, det_centroids[i]))
                    pid_to_kp[pid] = kp_by_det.get(best_i)

            # Annotate each tracked person
            for pid, info in self.tracker.objects.items():
                bbox = info["bbox"]
                x1, y1, x2, y2 = bbox
                cx, cy = info["centroid"]
                lm     = pid_to_kp.get(pid)

                # Category via face/pose estimator
                category, _ = self.age_estimator.classify(pid, frame, bbox, lm)
                color        = self._COLORS.get(category, (200, 200, 200))

                # Skeleton
                if lm is not None:
                    draw_skeleton(annotated, lm, color, fw, fh)

                # Fall analysis
                action, confidence = "detected", 0.5
                if lm is not None:
                    action, confidence = self.fall_detector.analyze(pid, lm, fh)

                # Bounding box
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                label = f"{category} | {action} ({confidence:.0%})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1-lh-10), (x1+lw+6, y1), color, -1)
                cv2.putText(annotated, label, (x1+3, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if action in ("fall_detected", "lying_motionless", "inactivity"):
                    events.append({"event_type": action, "person_id": pid,
                                   "person_category": category, "confidence": confidence,
                                   "bbox": bbox, "camera_id": camera_id})

                if category == "child" and zones:
                    for zone in zones:
                        if self._point_in_polygon(cx, cy, zone["polygon"]):
                            events.append({"event_type": "zone_entry", "person_id": pid,
                                           "person_category": "child", "confidence": 0.95,
                                           "bbox": bbox, "camera_id": camera_id,
                                           "zone_name": zone["zone_name"]})

            # Draw zones
            for zone in (zones or []):
                pts = np.array(zone["polygon"], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0,0,255), 2)
                if len(pts):
                    cv2.putText(annotated, f"ZONE: {zone['zone_name']}",
                                (pts[0][0], pts[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # HUD
            hud = f"Persons: {len(self.tracker.objects)}  [{DEVICE.upper()}{'  FP16' if USE_FP16 else ''}  POSE]"
            cv2.putText(annotated, hud, (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(annotated, hud, (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

            outputs.append((annotated, events))

        return outputs

    @staticmethod
    def _point_in_polygon(x, y, polygon):
        n, inside, j = len(polygon), False, len(polygon)-1
        for i in range(n):
            xi, yi = polygon[i]; xj, yj = polygon[j]
            if ((yi>y) != (yj>y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi):
                inside = not inside
            j = i
        return inside