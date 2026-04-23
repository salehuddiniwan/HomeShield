"""
HomeShield Camera Manager — handles multi-camera RTSP streaming with threaded capture.
"""
import cv2
import threading
import time
from config import Config


class CameraStream:
    """Threaded camera capture to avoid blocking the main pipeline."""

    def __init__(self, camera_id, name, url, location=""):
        self.camera_id = camera_id
        self.name = name
        self.url = url
        self.location = location
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        self.fps = 0
        self._frame_count = 0
        self._fps_time = time.time()

    def start(self):
        src = int(self.url) if self.url.isdigit() else self.url
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera: {self.name} ({self.url})")
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        print(f"[INFO] Camera started: {self.name} ({self.url})")
        return True

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = cv2.resize(
                        frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
                    )
                    self._frame_count += 1
                    elapsed = time.time() - self._fps_time
                    if elapsed >= 1.0:
                        self.fps = self._frame_count / elapsed
                        self._frame_count = 0
                        self._fps_time = time.time()
            time.sleep(1.0 / 30)  # cap at 30fps read rate

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print(f"[INFO] Camera stopped: {self.name}")

    @property
    def is_active(self):
        return self.running and self.grabbed


class CameraManager:
    """Manages multiple camera streams."""

    def __init__(self):
        self.cameras = {}  # camera_id -> CameraStream

    def add_camera(self, camera_id, name, url, location=""):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
        stream = CameraStream(camera_id, name, url, location)
        success = stream.start()
        if success:
            self.cameras[camera_id] = stream
        return success

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]

    def get_frame(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id].read()
        return False, None

    def get_all_active(self):
        return {cid: cam for cid, cam in self.cameras.items() if cam.is_active}

    def get_status(self):
        return {
            cid: {
                "name": cam.name,
                "location": cam.location,
                "active": cam.is_active,
                "fps": round(cam.fps, 1),
            }
            for cid, cam in self.cameras.items()
        }

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()
        self.cameras.clear()
