"""YOLOv8 ONNX inference + PPE violation detection.

Loads ONNX weights from HuggingFace Hub on first init (cached under
~/.cache/huggingface/hub/), runs CPU inference via onnxruntime, applies NMS,
and pairs detected PPE-absence classes (NO-Hardhat, NO-Safety Vest, NO-Mask)
with the nearest Person bbox to produce violations.

Brief reference:
    Layer 1 — Computer Vision (core)
    Violation rule: requires BOTH a person AND missing PPE in the same region.
"""

from __future__ import annotations

import ast
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
HF_REPO = "ayushgupta7777/safetyvision-yolov8"
ONNX_FILENAME = "best.onnx"
ONNX_DATA_FILENAME = "best.onnx.data"  # external weight data; must ride along

IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45
PERSON_IOU_MIN = 0.05  # min IoU between NO-X and Person to count as violation

# Risk levels per violation class (from brief Layer 4 spec)
RISK_LEVELS: dict[str, str] = {
    "NO-Hardhat": "HIGH",
    "NO-Safety Vest": "HIGH",
    "NO-Mask": "MEDIUM",
}
VIOLATION_CLASSES = set(RISK_LEVELS.keys())
PERSON_CLASS = "Person"


# ─── Result types ───────────────────────────────────────────────────────────
@dataclass
class Detection:
    cls: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image coords


@dataclass
class Violation:
    type: str
    risk_level: str
    confidence: float
    bbox: tuple[float, float, float, float]
    person_bbox: tuple[float, float, float, float] | None


@dataclass
class DetectionResult:
    detections: list[Detection]
    violations: list[Violation]
    image_shape: tuple[int, int]  # (h, w)
    inference_ms: float


# ─── Detector ───────────────────────────────────────────────────────────────
class PPEDetector:
    """YOLOv8 ONNX detector. First instance downloads weights from HF Hub."""

    _instance: PPEDetector | None = None

    def __init__(
        self,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        hf_repo: str = HF_REPO,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Pull both .onnx and its external-data file (must be in the same dir)
        onnx_path = hf_hub_download(repo_id=hf_repo, filename=ONNX_FILENAME)
        try:
            hf_hub_download(repo_id=hf_repo, filename=ONNX_DATA_FILENAME)
        except Exception:
            logger.warning(
                "No %s found in %s — assuming weights are inline", ONNX_DATA_FILENAME, hf_repo
            )

        logger.info("Loading ONNX model from %s", onnx_path)
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # ultralytics embeds class names in ONNX metadata as a Python-dict string
        meta = self.session.get_modelmeta().custom_metadata_map
        names_raw = meta.get("names", "{}")
        self.class_names: dict[int, str] = ast.literal_eval(names_raw)
        logger.info(
            "Loaded %d classes: %s",
            len(self.class_names),
            list(self.class_names.values()),
        )

    @classmethod
    def get(cls) -> PPEDetector:
        """Singleton accessor for Lambda warm-container reuse."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── preprocessing ───────────────────────────────────────────────────────
    def _letterbox(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize preserving aspect ratio, pad to IMG_SIZE x IMG_SIZE with gray."""
        h, w = img.shape[:2]
        scale = min(IMG_SIZE / h, IMG_SIZE / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = IMG_SIZE - new_h, IMG_SIZE - new_w
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return padded, scale, (left, top)

    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        padded, scale, pad = self._letterbox(img_rgb)
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, :]  # HWC → NCHW
        return np.ascontiguousarray(tensor), scale, pad

    # ── postprocessing ──────────────────────────────────────────────────────
    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> list[Detection]:
        # YOLOv8 ONNX output: (1, 4 + num_classes, N)
        preds = output[0].T  # → (N, 4 + num_classes)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        # Confidence filter
        keep = confidences > self.conf_threshold
        if not keep.any():
            return []
        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # xywh (center) → xyxy
        cx, cy, w, h = boxes_xywh.T
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS via OpenCV (expects [x, y, w, h])
        boxes_for_nms = np.stack([x1, y1, w, h], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(
            boxes_for_nms, confidences.tolist(),
            self.conf_threshold, self.iou_threshold,
        )
        if len(idxs) == 0:
            return []
        idxs_arr = np.array(idxs).flatten()

        # Undo letterbox: subtract pad, divide by scale, clamp to image bounds
        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape
        results: list[Detection] = []
        for i in idxs_arr:
            bx1 = max(0.0, (boxes_xyxy[i, 0] - pad_x) / scale)
            by1 = max(0.0, (boxes_xyxy[i, 1] - pad_y) / scale)
            bx2 = min(float(orig_w), (boxes_xyxy[i, 2] - pad_x) / scale)
            by2 = min(float(orig_h), (boxes_xyxy[i, 3] - pad_y) / scale)
            results.append(Detection(
                cls=self.class_names[int(class_ids[i])],
                confidence=float(confidences[i]),
                bbox=(bx1, by1, bx2, by2),
            ))
        return results

    # ── violation logic ─────────────────────────────────────────────────────
    @staticmethod
    def _iou(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _detect_violations(self, dets: list[Detection]) -> list[Violation]:
        """Pair each NO-X detection with the best-overlapping Person bbox.

        Brief rule: violation requires BOTH a person AND missing PPE. If a
        NO-X has no nearby person, drop it as a likely false positive.
        """
        persons = [d for d in dets if d.cls == PERSON_CLASS]
        if not persons:
            return []

        violations: list[Violation] = []
        for d in dets:
            if d.cls not in VIOLATION_CLASSES:
                continue
            best_iou, best_person = 0.0, None
            for p in persons:
                iou = self._iou(d.bbox, p.bbox)
                if iou > best_iou:
                    best_iou, best_person = iou, p
            if best_iou >= PERSON_IOU_MIN and best_person is not None:
                violations.append(Violation(
                    type=d.cls,
                    risk_level=RISK_LEVELS[d.cls],
                    confidence=d.confidence,
                    bbox=d.bbox,
                    person_bbox=best_person.bbox,
                ))
        return violations

    # ── public API ──────────────────────────────────────────────────────────
    def predict(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Run detection. Accepts BGR ndarray or image path."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H, W, 3), got shape {img.shape}")

        h, w = img.shape[:2]
        tensor, scale, pad = self._preprocess(img)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(outputs[0], scale, pad, (h, w))
        violations = self._detect_violations(detections)

        return DetectionResult(
            detections=detections,
            violations=violations,
            image_shape=(h, w),
            inference_ms=inference_ms,
        )


# ─── Annotation helper ──────────────────────────────────────────────────────
def draw_annotations(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw bboxes on a BGR image copy.

    Red = violation (NO-X paired with a person), Yellow = person,
    Green = positive PPE / other.
    """
    out = image.copy()
    violation_bbox_set = {tuple(v.bbox) for v in result.violations}
    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        if tuple(det.bbox) in violation_bbox_set:
            color = (0, 0, 255)       # red — BGR
        elif det.cls == PERSON_CLASS:
            color = (0, 255, 255)     # yellow — BGR
        else:
            color = (0, 255, 0)       # green — BGR

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return out
