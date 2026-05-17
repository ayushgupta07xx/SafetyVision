"""Tests for core/detector.py mechanics (IoU, preprocess, postprocess,
drawing, constants) and core/rag.py::format_chunks_for_prompt.

Violation-pairing logic is in tests/test_violation.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.detector import (
    RISK_LEVELS,
    VIOLATION_CLASSES,
    Detection,
    DetectionResult,
    PPEDetector,
    Violation,
    draw_annotations,
)


# ─── IoU math ───────────────────────────────────────────────────────────────
class TestIoU:
    def test_identical_bboxes(self):
        bbox = (10.0, 10.0, 100.0, 100.0)
        assert PPEDetector._iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (100.0, 100.0, 110.0, 110.0)
        assert PPEDetector._iou(a, b) == 0.0

    def test_partial_overlap(self):
        # 10×10 + 10×10 sharing a 5×5 corner: inter=25, union=175
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        assert PPEDetector._iou(a, b) == pytest.approx(25 / 175)

    def test_zero_area_box_returns_zero(self):
        a = (0.0, 0.0, 0.0, 0.0)
        b = (0.0, 0.0, 10.0, 10.0)
        assert PPEDetector._iou(a, b) == 0.0

    def test_one_contained_in_other(self):
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (10.0, 10.0, 20.0, 20.0)
        # inter=100, union=10000+100-100=10000 → 0.01
        assert PPEDetector._iou(outer, inner) == pytest.approx(0.01)


# ─── Constants sanity ───────────────────────────────────────────────────────
class TestConstants:
    def test_violation_classes_have_risk_levels(self):
        for cls in VIOLATION_CLASSES:
            assert cls in RISK_LEVELS

    def test_risk_levels_are_valid_strings(self):
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for cls, risk in RISK_LEVELS.items():
            assert risk in valid, f"{cls}: {risk}"

    def test_known_classes_present(self):
        assert "NO-Hardhat" in VIOLATION_CLASSES
        assert "NO-Safety Vest" in VIOLATION_CLASSES
        assert "NO-Mask" in VIOLATION_CLASSES


# ─── Dataclasses ────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_detection_construction(self):
        d = Detection(cls="Person", confidence=0.91, bbox=(0.0, 0.0, 100.0, 200.0))
        assert d.cls == "Person"
        assert d.confidence == pytest.approx(0.91)
        assert d.bbox == (0.0, 0.0, 100.0, 200.0)

    def test_detection_result_construction(self):
        r = DetectionResult(
            detections=[], violations=[], image_shape=(100, 200), inference_ms=12.3,
        )
        assert r.image_shape == (100, 200)
        assert r.inference_ms == pytest.approx(12.3)


# ─── draw_annotations ───────────────────────────────────────────────────────
class TestDrawAnnotations:
    def test_returns_same_shape(self, sample_bgr):
        result = DetectionResult(
            detections=[], violations=[], image_shape=(100, 200), inference_ms=0.0,
        )
        out = draw_annotations(sample_bgr, result)
        assert out.shape == sample_bgr.shape
        assert out.dtype == sample_bgr.dtype

    def test_does_not_mutate_input(self, sample_bgr):
        result = DetectionResult(
            detections=[Detection(cls="Person", confidence=0.9, bbox=(10, 10, 50, 50))],
            violations=[], image_shape=(100, 200), inference_ms=0.0,
        )
        original = sample_bgr.copy()
        draw_annotations(sample_bgr, result)
        assert np.array_equal(sample_bgr, original)

    def test_violation_box_is_drawn(self, sample_bgr):
        bbox = (10.0, 10.0, 50.0, 50.0)
        det = Detection(cls="NO-Hardhat", confidence=0.9, bbox=bbox)
        viol = Violation(
            type="NO-Hardhat", risk_level="HIGH", confidence=0.9,
            bbox=bbox, person_bbox=None,
        )
        result = DetectionResult(
            detections=[det], violations=[viol],
            image_shape=(100, 200), inference_ms=0.0,
        )
        out = draw_annotations(sample_bgr, result)
        assert not np.array_equal(out, sample_bgr)


# ─── Preprocess / letterbox shapes ──────────────────────────────────────────
class TestPreprocess:
    def test_preprocess_shape_and_dtype(self, detector_instance, sample_bgr):
        tensor, scale, pad = detector_instance._preprocess(sample_bgr)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= float(tensor.max()) <= 1.0
        assert isinstance(scale, float)
        assert isinstance(pad, tuple) and len(pad) == 2

    def test_letterbox_preserves_aspect(self, detector_instance):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, scale, pad = detector_instance._letterbox(img)
        assert padded.shape == (640, 640, 3)
        # scale = min(640/100, 640/200) = 3.2
        assert scale == pytest.approx(3.2)

    def test_letterbox_square_input(self, detector_instance):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, pad = detector_instance._letterbox(img)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert pad == (0, 0)


# ─── _postprocess: YOLOv8 ONNX output → list[Detection] ─────────────────────
class TestPostprocess:
    """Exercise NMS, confidence filtering, and bbox unprojection on a
    synthetic YOLOv8 output tensor (1, 4 + num_classes, num_anchors)."""

    @staticmethod
    def _make_yolo_output(num_classes: int, anchors: list[tuple]) -> np.ndarray:
        """Build a synthetic ONNX output.

        Each anchor is (cx, cy, w, h, class_id, confidence) in input-tensor
        coordinates (the letterboxed 640×640 space). Class scores for other
        classes are left at 0, so argmax picks `class_id` and max picks
        `confidence`.
        """
        n = max(len(anchors), 1)
        out = np.zeros((1, 4 + num_classes, n), dtype=np.float32)
        for i, (cx, cy, w, h, cid, conf) in enumerate(anchors):
            out[0, 0, i] = cx
            out[0, 1, i] = cy
            out[0, 2, i] = w
            out[0, 3, i] = h
            out[0, 4 + cid, i] = conf
        return out

    def test_all_below_threshold_returns_empty(self, detector_instance):
        # Single anchor with conf 0.2 < 0.40 threshold
        out = self._make_yolo_output(7, [(100, 100, 50, 50, 0, 0.2)])
        result = detector_instance._postprocess(
            out, scale=1.0, pad=(0, 0), orig_shape=(640, 640),
        )
        assert result == []

    def test_single_high_confidence_detection(self, detector_instance):
        out = self._make_yolo_output(7, [(320, 320, 100, 100, 0, 0.9)])
        result = detector_instance._postprocess(
            out, scale=1.0, pad=(0, 0), orig_shape=(640, 640),
        )
        assert len(result) == 1
        assert result[0].cls == "Person"            # class_id 0 in the fixture
        assert result[0].confidence == pytest.approx(0.9, abs=0.05)

    def test_class_id_mapped_via_class_names(self, detector_instance):
        # class_id 2 → NO-Hardhat in fixture
        out = self._make_yolo_output(7, [(320, 320, 100, 100, 2, 0.85)])
        result = detector_instance._postprocess(
            out, scale=1.0, pad=(0, 0), orig_shape=(640, 640),
        )
        assert result[0].cls == "NO-Hardhat"

    def test_bbox_unprojected_to_original_coords(self, detector_instance):
        # A 100×200 image letterboxed at IMG_SIZE=640 has scale=3.2, pad=(0,160).
        # An anchor at (cx=320, cy=320, w=100, h=100) → xyxy (270, 270, 370, 370)
        # Undoing: x' = (x - pad_x) / scale,  y' = (y - pad_y) / scale
        #   x1 = (270 - 0) / 3.2 = 84.375
        #   y1 = (270 - 160) / 3.2 = 34.375
        #   x2 = (370 - 0) / 3.2 = 115.625
        #   y2 = (370 - 160) / 3.2 = 65.625
        out = self._make_yolo_output(7, [(320, 320, 100, 100, 0, 0.9)])
        result = detector_instance._postprocess(
            out, scale=3.2, pad=(0, 160), orig_shape=(100, 200),
        )
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x1 == pytest.approx(84.375, abs=1.0)
        assert y1 == pytest.approx(34.375, abs=1.0)
        assert x2 == pytest.approx(115.625, abs=1.0)
        assert y2 == pytest.approx(65.625, abs=1.0)

    def test_bbox_clamped_to_image_bounds(self, detector_instance):
        # Anchor extends well outside image; coordinates must clamp to [0, w/h].
        out = self._make_yolo_output(7, [(50, 50, 200, 200, 0, 0.9)])
        result = detector_instance._postprocess(
            out, scale=1.0, pad=(0, 0), orig_shape=(100, 100),
        )
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x1 >= 0.0 and y1 >= 0.0
        assert x2 <= 100.0 and y2 <= 100.0

    def test_multiple_classes_kept(self, detector_instance):
        # Two well-separated anchors of different classes → both survive NMS
        out = self._make_yolo_output(7, [
            (100, 100, 50, 50, 0, 0.9),    # Person
            (500, 500, 50, 50, 2, 0.85),   # NO-Hardhat
        ])
        result = detector_instance._postprocess(
            out, scale=1.0, pad=(0, 0), orig_shape=(640, 640),
        )
        classes = {d.cls for d in result}
        assert classes == {"Person", "NO-Hardhat"}

    def test_nms_dedupes_overlapping_same_class(self, detector_instance):
        # Two near-identical Person anchors — NMS drops the lower-confidence one
        out = self._make_yolo_output(7, [
            (100, 100, 50, 50, 0, 0.90),
            (102, 102, 50, 50, 0, 0.85),
        ])
        result = detector_instance._postprocess(
            out, scale=1.0, pad=(0, 0), orig_shape=(640, 640),
        )
        assert len(result) == 1
        # Higher-confidence anchor wins
        assert result[0].confidence == pytest.approx(0.90, abs=0.05)


# ─── core.rag.format_chunks_for_prompt ──────────────────────────────────────
class TestFormatChunksForPrompt:
    def test_empty_chunks(self):
        from core.rag import format_chunks_for_prompt
        out = format_chunks_for_prompt([])
        assert "no relevant" in out.lower()

    def test_includes_source_score_and_text(self):
        from core.rag import RetrievedChunk, format_chunks_for_prompt
        chunks = [
            RetrievedChunk(
                text="Hard hats must be worn at all times.",
                source="1926_100.txt",
                chunk_idx=0,
                score=0.87,
            ),
        ]
        out = format_chunks_for_prompt(chunks)
        assert "1926.100" in out
        assert "0.87" in out
        assert "Hard hats must be worn" in out

    def test_multiple_chunks_separated(self):
        from core.rag import RetrievedChunk, format_chunks_for_prompt
        chunks = [
            RetrievedChunk(text="A", source="1926_100.txt", chunk_idx=0, score=0.9),
            RetrievedChunk(text="B", source="1910_134.txt", chunk_idx=1, score=0.8),
        ]
        out = format_chunks_for_prompt(chunks)
        assert "1926.100" in out
        assert "1910.134" in out
