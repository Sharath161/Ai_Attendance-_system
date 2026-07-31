"""Self-contained face embedding engine — detection, alignment, recognition.

Domain-agnostic: knows nothing about attendance. Given an image it returns
face detections and 512-d L2-normalised ArcFace embeddings. Supports:
  * flip test-time augmentation (accuracy)
  * INT8-quantized recognition model + thread pinning (low-resource / edge)

Detection : YuNet (OpenCV Zoo ONNX, 5-point landmarks)
Recognition: ArcFace (InsightFace w600k_*, 512-d)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ArcFace canonical 112x112 landmark template (left_eye, right_eye, nose, l_mouth, r_mouth)
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _yunet_to_arcface_kps(face_row: np.ndarray) -> np.ndarray:
    return np.array(
        [[face_row[6], face_row[7]], [face_row[4], face_row[5]],
         [face_row[8], face_row[9]], [face_row[12], face_row[13]],
         [face_row[10], face_row[11]]], dtype=np.float32)


def _align(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(image, (112, 112))
    return cv2.warpAffine(image, M, (112, 112), borderValue=0)


def _preprocess(face_112: np.ndarray) -> np.ndarray:
    face = (face_112.astype(np.float32) - 127.5) / 128.0
    return face.transpose(2, 0, 1)[np.newaxis, :]


@dataclass
class Detection:
    bbox: dict           # {x, y, w, h}
    confidence: float
    embedding: np.ndarray | None   # 512-d L2-normalised


class FaceEngine:
    def __init__(
        self,
        detection_model: str | Path,
        recognition_model: str | Path,
        detection_score_threshold: float = 0.60,
        intra_op_threads: int = 0,
        enable_tta: bool = False,
    ) -> None:
        detection_model = Path(detection_model)
        recognition_model = Path(recognition_model)
        for label, p in (("detection", detection_model), ("recognition", recognition_model)):
            if not p.exists():
                raise FileNotFoundError(f"Missing {label} model: {p}")

        self.enable_tta = enable_tta
        self.recognition_model_name = recognition_model.name

        self._detector = cv2.FaceDetectorYN.create(
            model=str(detection_model), config="", input_size=(320, 320),
            score_threshold=detection_score_threshold, nms_threshold=0.3, top_k=5000)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        so.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(recognition_model), sess_options=so, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    # ── low-level ────────────────────────────────────────────────────────────
    def _embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        raw = self._session.run(None, {self._input: _preprocess(aligned)})[0][0]
        return _l2(np.asarray(raw, dtype=np.float32).flatten())

    def _embed_face(self, image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        aligned = _align(image, kps)
        emb = self._embed_aligned(aligned)
        if self.enable_tta:
            emb_f = self._embed_aligned(cv2.flip(aligned, 1))
            emb = _l2((emb + emb_f) / 2.0)
        return emb

    # ── public ───────────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray, embed: bool = True, max_faces: int = 10) -> list[Detection]:
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return []
        faces = sorted(faces, key=lambda f: -f[-1])[:max_faces]
        out = []
        for f in faces:
            bbox = {"x": int(f[0]), "y": int(f[1]), "w": int(f[2]), "h": int(f[3])}
            emb = self._embed_face(image, _yunet_to_arcface_kps(f)) if embed else None
            out.append(Detection(bbox=bbox, confidence=float(f[-1]), embedding=emb))
        return out

    def embed_primary(self, image: np.ndarray) -> Detection | None:
        """Embed the single most-confident face, or None (and face_count via .detect)."""
        dets = self.detect(image, embed=True, max_faces=10)
        if not dets:
            return None
        return dets[0]

    def decode(self, data: bytes) -> np.ndarray | None:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
