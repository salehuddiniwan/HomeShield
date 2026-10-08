"""Face recognition using InsightFace.

Provides:
  * Face detection + 512-d ArcFace embedding (+ age / gender estimates)
  * Gallery matching via cosine similarity

If InsightFace isn't installed the module still imports cleanly and
`FaceEngine.available` is False — all callers must check.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import warnings
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# InsightFace 0.7.3 calls np.linalg.lstsq without rcond and skimage's
# deprecated SimilarityTransform.estimate. Both still work; the warnings are
# only noise in the console.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.")


def _try_import_face_analysis():
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        return FaceAnalysis
    except Exception as e:
        log.warning("insightface not loadable: %s", e)
        return None


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FaceEngine:
    """Thread-safe face detector + ArcFace embedder."""

    # Cosine similarity for "this is person X".
    # InsightFace ArcFace: 0.40 is permissive, 0.50 is strict.
    DEFAULT_THRESHOLD = 0.42

    def __init__(self, model_name: str = "buffalo_l",
                 det_size: tuple[int, int] = (640, 640),
                 prefer_gpu: bool = True):
        self.available: bool = False
        self.last_error: Optional[str] = None
        self.device: str = "none"     # "cuda" | "cpu" once loaded
        self._lock = threading.Lock()
        self._app = None

        use_gpu = prefer_gpu and _cuda_available()
        self._try_face_analysis(model_name, det_size, use_gpu)

    # ---- init helpers ---------------------------------------------------

    def _try_face_analysis(self, model_name: str,
                           det_size: tuple[int, int], use_gpu: bool) -> bool:
        FaceAnalysis = _try_import_face_analysis()
        if FaceAnalysis is None:
            self.last_error = (
                "insightface not installed. "
                "Try: pip install insightface onnxruntime-gpu  "
                "(or use the wheel under Face_Detection/)"
            )
            return False
        for gpu in ([True, False] if use_gpu else [False]):
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if gpu else ["CPUExecutionProvider"])
            try:
                # InsightFace print()s a "find model" / "Applied providers"
                # line per model; keep them for --debug only.
                chatter = io.StringIO()
                with contextlib.redirect_stdout(chatter):
                    app = FaceAnalysis(name=model_name, providers=providers)
                    app.prepare(ctx_id=0 if gpu else -1, det_size=det_size)
                for line in chatter.getvalue().splitlines():
                    if line.strip():
                        log.debug("insightface: %s", line)
                self._app = app
                self.available = True
                self.last_error = None
                self.device = self._session_device(app)
                log.info("FaceAnalysis ready on %s (det_size=%s)", self.device, det_size)
                if gpu and self.device != "cuda":
                    # ONNX Runtime drops to CPU with only a console message
                    # when its CUDA build doesn't match the CUDA libraries
                    # PyTorch ships (e.g. onnxruntime-gpu 1.27+ needs CUDA 13).
                    import onnxruntime
                    import torch
                    log.warning(
                        "face recognition is running on the CPU: onnxruntime-gpu %s "
                        "could not use CUDA with PyTorch %s (CUDA %s). Install the "
                        "matching build: pip install -e .[cuda13] for CUDA 13, "
                        ".[cuda12] for CUDA 12.",
                        onnxruntime.__version__, torch.__version__, torch.version.cuda)
                return True
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("FaceAnalysis init failed (gpu=%s): %s", gpu, e)
        return False

    @staticmethod
    def _session_device(app) -> str:
        """'cuda' if the detector's ONNX session actually runs on CUDA."""
        try:
            providers = app.models["detection"].session.get_providers()
        except Exception:
            return "unknown"
        return "cuda" if "CUDAExecutionProvider" in providers else "cpu"

    # ---- public API -----------------------------------------------------

    def detect(self, frame_bgr) -> list[dict[str, Any]]:
        if (not self.available or self._app is None
                or frame_bgr is None or frame_bgr.size == 0):
            return []
        with self._lock:
            try:
                faces = self._app.get(frame_bgr)
            except Exception as e:
                self.last_error = f"detect: {e}"
                return []
        return [_face_obj_to_dict(f) for f in faces]

    def best_face(self, frame_bgr) -> Optional[dict[str, Any]]:
        faces = self.detect(frame_bgr)
        return max(faces, key=lambda f: f["w"] * f["h"]) if faces else None


# ---- helpers --------------------------------------------------------------

def _face_obj_to_dict(face) -> dict[str, Any]:
    """Convert an InsightFace `Face` proxy to the homeshield dict shape."""
    bbox = np.asarray(face.bbox, dtype=float)
    emb = getattr(face, "normed_embedding", None)
    if emb is None and hasattr(face, "embedding"):
        e = np.asarray(face.embedding, dtype=np.float32)
        emb = e / (np.linalg.norm(e) or 1.0)
    if emb is not None:
        emb = np.asarray(emb, dtype=np.float32)
    return {
        "x": float(bbox[0]),
        "y": float(bbox[1]),
        "w": float(bbox[2] - bbox[0]),
        "h": float(bbox[3] - bbox[1]),
        "age": int(face.age) if getattr(face, "age", None) is not None else None,
        "gender": int(face.gender) if getattr(face, "gender", None) is not None else None,
        "det_score": float(getattr(face, "det_score", 0.0)),
        "embedding": emb,
    }


def is_good_face(face: dict[str, Any], *, min_size: float = 40.0,
                 min_det_score: float = 0.6) -> bool:
    """True when a face is large and confident enough to trust its embedding.

    ArcFace embeddings from tiny, blurred or profile faces land far from the
    person's enrolled embedding, so matching them produces false "unknown"
    results (spurious intruder alerts) rather than real ones.
    """
    return (face.get("embedding") is not None
            and float(face.get("det_score", 0.0)) >= min_det_score
            and min(float(face.get("w", 0.0)), float(face.get("h", 0.0))) >= min_size)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a, b = _unit(a), _unit(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def best_match(query, gallery, threshold: float = FaceEngine.DEFAULT_THRESHOLD):
    """gallery: list of (id, embedding). Returns (best_id_or_None, score).

    Vectorised: stacks the gallery into one matrix and does a single
    matrix-vector dot, which is ~10x faster than per-loop cosine for a
    gallery of 10+ persons.
    """
    if not gallery or query is None:
        return None, 0.0
    valid = [(pid, _unit(emb)) for pid, emb in gallery if emb is not None]
    if not valid:
        return None, 0.0
    ids = [pid for pid, _ in valid]
    mat = np.stack([emb for _, emb in valid])  # (N, D), unit-normalised
    sims = mat @ _unit(query)                  # (N,)
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    return (ids[idx], score) if score >= threshold else (None, score)
