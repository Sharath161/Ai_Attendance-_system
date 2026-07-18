"""Face pipeline: YuNet detection + ArcFace ONNX recognition.

Detection  — YuNet (OpenCV Zoo ONNX), produces accurate 5-point landmarks
Embedding  — ArcFace MobileNet (InsightFace w600k_mbf.onnx), 512-d L2-normalised
Similarity — cosine (dot product of L2-normalised vectors)

This replaces the original YuNet + SFace (128-d) adapter with ArcFace (512-d),
giving significantly better inter-person separation and a higher match threshold.

Download both models with:
    python -m worker.download_models
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from core.config import Settings
from core.math_utils import l2_normalize

# ArcFace 112×112 canonical landmark positions.
# Order: left_eye, right_eye, nose, left_mouth, right_mouth.
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _yunet_to_arcface_kps(face_row: np.ndarray) -> np.ndarray:
    """Reorder YuNet landmark columns to ArcFace's expected order.

    YuNet row layout (indices 4–13):
        4,5   = right_eye
        6,7   = left_eye
        8,9   = nose
        10,11 = right_mouth
        12,13 = left_mouth

    ArcFace expects: left_eye, right_eye, nose, left_mouth, right_mouth.
    """
    return np.array(
        [
            [face_row[6],  face_row[7]],   # left_eye
            [face_row[4],  face_row[5]],   # right_eye
            [face_row[8],  face_row[9]],   # nose
            [face_row[12], face_row[13]],  # left_mouth
            [face_row[10], face_row[11]],  # right_mouth
        ],
        dtype=np.float32,
    )


def _align_face(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Affine-warp the face region to ArcFace's 112×112 canonical pose."""
    M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(image, (112, 112))
    return cv2.warpAffine(image, M, (112, 112), borderValue=0)


def _preprocess_arcface(face_112: np.ndarray) -> np.ndarray:
    """Normalise to [-1, 1] and reshape to ONNX input [1, 3, 112, 112]."""
    face = face_112.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)   # HWC → CHW
    return face[np.newaxis, :]        # add batch dimension


class FaceEmbeddingModel:
    """YuNet detection + ArcFace ONNX embedding pipeline.

    Keeps the same public interface as the previous adapter (embed_primary_face,
    embed_array, detect_and_embed, embed_image, from_settings) so no call-sites
    in the API or worker need updating.

    Key upgrade over the original YuNet + SFace pipeline:
    - ArcFace produces 512-d embeddings (vs SFace's 128-d)
    - Much better inter-person separation → fewer false matches
    - Proper landmark-based alignment → robust to pose/lighting variation
    """

    def __init__(
        self,
        model_version: str,
        detection_model_path: Path,
        recognition_model_path: Path,
        embedding_dimensions: int = 512,
        detection_score_threshold: float = 0.60,
    ) -> None:
        detection_model_path = Path(detection_model_path)
        recognition_model_path = Path(recognition_model_path)
        for label, path in (
            ("detection", detection_model_path),
            ("recognition", recognition_model_path),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {label} model at {path}. "
                    "Run `python -m worker.download_models` to fetch the models."
                )

        self.model_version = model_version
        self.embedding_dimensions = embedding_dimensions

        self._detector = cv2.FaceDetectorYN.create(
            model=str(detection_model_path),
            config="",
            input_size=(320, 320),
            score_threshold=detection_score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._session = ort.InferenceSession(
            str(recognition_model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    @classmethod
    def from_settings(cls, settings: Settings) -> "FaceEmbeddingModel":
        return cls(
            model_version=settings.worker_model_version,
            detection_model_path=settings.face_detection_model_path,
            recognition_model_path=settings.face_recognition_model_path,
            embedding_dimensions=settings.embedding_dimensions,
            detection_score_threshold=settings.detection_score_threshold,
        )

    # ── public interface ──────────────────────────────────────────────────────

    def detect_and_embed(
        self, image: np.ndarray
    ) -> tuple[np.ndarray | None, int, dict | None]:
        """Detect faces and embed the primary one.

        Returns ``(embedding, face_count, bbox)`` where ``bbox`` is
        ``{"x", "y", "w", "h"}`` in original-image pixel coordinates, or
        ``None`` when no face is found.
        """
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return None, 0, None

        face_count = len(faces)
        # Primary face = highest YuNet confidence score (last column)
        primary = max(faces, key=lambda f: f[-1])

        bbox = {
            "x": int(primary[0]),
            "y": int(primary[1]),
            "w": int(primary[2]),
            "h": int(primary[3]),
        }

        # Reorder YuNet landmarks → ArcFace convention, then align to 112×112
        kps = _yunet_to_arcface_kps(primary)
        aligned = _align_face(image, kps)

        inp = _preprocess_arcface(aligned)
        raw = self._session.run(None, {self._input_name: inp})[0][0]
        embedding = l2_normalize(np.asarray(raw, dtype=np.float32).flatten())
        return embedding, face_count, bbox

    def embed_primary_face(self, image_path: Path) -> tuple[np.ndarray | None, int]:
        """Return the embedding of the most prominent face and the total face count."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, 0
        embedding, face_count, _ = self.detect_and_embed(image)
        return embedding, face_count

    def embed_array(self, image: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Embed from an in-memory BGR frame."""
        embedding, face_count, _ = self.detect_and_embed(image)
        return embedding, face_count

    def embed_image(self, image_path: Path) -> np.ndarray | None:
        """Backward-compatible single-embedding API used by registration."""
        embedding, _ = self.embed_primary_face(image_path)
        return embedding
