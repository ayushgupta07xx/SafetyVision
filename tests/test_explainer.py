"""Tests for core/explainer.py — non-torch paths.

What's covered:
    - _bbox_mask_letterboxed, _apply_bbox_mask, _heatmap_overlay (pure numpy/cv2)
    - _png_b64, _fig_to_b64 (encoding helpers)
    - _preprocess_torch (creates a torch tensor but no model load)
    - annotations_b64 (pure: draw + encode, no torch model)
    - explain_result (logic + dispatch; torch heatmap funcs mocked)

What's NOT covered (and is marked `# pragma: no cover` in core/explainer.py):
    - _TorchModelCache, _YoloClassTarget, _YoloScalarWrapper
    - gradcam_heatmap_b64, shap_attribution_b64
    These require a real YOLO model with the expected architecture to compute
    meaningful gradients; mocking them produces tautological tests.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from core.explainer import (
    EXPLAINER_IMG_SIZE,
    Explanation,
    _apply_bbox_mask,
    _bbox_mask_letterboxed,
    _fig_to_b64,
    _heatmap_overlay,
    _png_b64,
    _preprocess_torch,
    annotations_b64,
)


# ─── _bbox_mask_letterboxed ─────────────────────────────────────────────────
class TestBboxMaskLetterboxed:
    def test_shape_and_dtype(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        mask = _bbox_mask_letterboxed(img, [(0.0, 0.0, 50.0, 50.0)])
        assert mask.shape == (EXPLAINER_IMG_SIZE, EXPLAINER_IMG_SIZE)
        assert mask.dtype == np.float32

    def test_empty_bboxes_yields_zero_mask(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        mask = _bbox_mask_letterboxed(img, [])
        assert mask.sum() == 0.0

    def test_full_image_bbox_fills_letterbox_region(self):
        # 100×200 image at size=320:
        #   scale = min(320/100, 320/200) = 1.6
        #   new_h, new_w = 160, 320
        #   pad_y = (320-160)//2 = 80, pad_x = 0
        # So a full-image bbox maps to rows 80–240, cols 0–320.
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        mask = _bbox_mask_letterboxed(img, [(0.0, 0.0, 200.0, 100.0)])
        assert mask[80:240, :].mean() == pytest.approx(1.0)
        assert mask[0:80, :].sum() == 0.0
        assert mask[240:, :].sum() == 0.0

    def test_oversized_bbox_clamps_without_error(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = _bbox_mask_letterboxed(img, [(-1000.0, -1000.0, 9999.0, 9999.0)])
        assert mask.shape == (EXPLAINER_IMG_SIZE, EXPLAINER_IMG_SIZE)
        assert mask.max() == 1.0

    def test_multiple_bboxes_union(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Two disjoint 10×10 bboxes — both should appear in mask
        mask = _bbox_mask_letterboxed(img, [(0, 0, 10, 10), (50, 50, 60, 60)])
        assert mask.sum() > 0


# ─── _apply_bbox_mask ───────────────────────────────────────────────────────
class TestApplyBboxMask:
    def test_all_one_mask_returns_overlay(self):
        size = EXPLAINER_IMG_SIZE
        overlay = np.full((size, size, 3), 255, dtype=np.uint8)
        letterboxed = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.ones((size, size), dtype=np.float32)
        out = _apply_bbox_mask(overlay, letterboxed, mask)
        assert (out == 255).all()

    def test_all_zero_mask_returns_letterboxed(self):
        size = EXPLAINER_IMG_SIZE
        overlay = np.full((size, size, 3), 255, dtype=np.uint8)
        letterboxed = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.float32)
        out = _apply_bbox_mask(overlay, letterboxed, mask)
        assert (out == 0).all()

    def test_split_mask_blends_per_region(self):
        size = EXPLAINER_IMG_SIZE
        overlay = np.full((size, size, 3), 200, dtype=np.uint8)
        letterboxed = np.full((size, size, 3), 100, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.float32)
        mask[: size // 2, :] = 1.0
        out = _apply_bbox_mask(overlay, letterboxed, mask)
        assert (out[: size // 2, :] == 200).all()
        assert (out[size // 2 :, :] == 100).all()


# ─── _heatmap_overlay ───────────────────────────────────────────────────────
class TestHeatmapOverlay:
    def test_zero_heatmap_preserves_original(self):
        size = EXPLAINER_IMG_SIZE
        letterboxed = np.full((size, size, 3), 128, dtype=np.uint8)
        heatmap = np.zeros((size, size), dtype=np.float32)
        out = _heatmap_overlay(letterboxed, heatmap)
        assert np.array_equal(out, letterboxed)

    def test_max_heatmap_modifies_pixels(self):
        size = EXPLAINER_IMG_SIZE
        letterboxed = np.full((size, size, 3), 100, dtype=np.uint8)
        heatmap = np.ones((size, size), dtype=np.float32)
        out = _heatmap_overlay(letterboxed, heatmap, alpha_max=0.6)
        assert not np.array_equal(out, letterboxed)
        assert out.shape == letterboxed.shape
        assert out.dtype == np.uint8

    def test_output_clipped_to_uint8_range(self):
        size = EXPLAINER_IMG_SIZE
        letterboxed = np.full((size, size, 3), 255, dtype=np.uint8)
        heatmap = np.ones((size, size), dtype=np.float32)
        out = _heatmap_overlay(letterboxed, heatmap)
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255


# ─── _png_b64 ───────────────────────────────────────────────────────────────
class TestPngB64:
    def test_emits_png_magic_bytes(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        b64 = _png_b64(img)
        assert isinstance(b64, str)
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_roundtrip_preserves_shape(self):
        img = np.array([[[0, 0, 255], [0, 255, 0]],
                        [[255, 0, 0], [128, 128, 128]]], dtype=np.uint8)
        b64 = _png_b64(img)
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded.shape == img.shape


# ─── _fig_to_b64 ────────────────────────────────────────────────────────────
class TestFigToB64:
    def test_emits_png_magic_bytes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        b64 = _fig_to_b64(fig)
        assert isinstance(b64, str)
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"


# ─── _preprocess_torch ──────────────────────────────────────────────────────
class TestPreprocessTorch:
    def test_tensor_shape_and_dtype(self):
        import torch
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, letterboxed = _preprocess_torch(img, requires_grad=False)
        assert tensor.shape == (1, 3, EXPLAINER_IMG_SIZE, EXPLAINER_IMG_SIZE)
        assert tensor.dtype == torch.float32
        assert letterboxed.shape == (EXPLAINER_IMG_SIZE, EXPLAINER_IMG_SIZE, 3)
        assert letterboxed.dtype == np.uint8

    def test_values_normalized(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        tensor, _ = _preprocess_torch(img, requires_grad=False)
        assert 0.0 <= float(tensor.max()) <= 1.0

    def test_requires_grad_flag(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        t_grad, _ = _preprocess_torch(img, requires_grad=True)
        t_no_grad, _ = _preprocess_torch(img, requires_grad=False)
        assert t_grad.requires_grad is True
        assert t_no_grad.requires_grad is False


# ─── annotations_b64 ────────────────────────────────────────────────────────
class TestAnnotationsB64:
    def test_emits_decodable_png(self, sample_bgr):
        from core.detector import Detection, DetectionResult
        result = DetectionResult(
            detections=[Detection(cls="Person", confidence=0.9, bbox=(10, 10, 50, 50))],
            violations=[],
            image_shape=(100, 200),
            inference_ms=0.0,
        )
        b64 = annotations_b64(sample_bgr, result)
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"


# ─── explain_result ─────────────────────────────────────────────────────────
class TestExplainResult:
    def test_returns_none_when_no_detections(self, sample_bgr):
        from core.detector import DetectionResult
        from core.explainer import explain_result
        result = DetectionResult(
            detections=[], violations=[],
            image_shape=(100, 200), inference_ms=0.0,
        )
        assert explain_result(sample_bgr, result) is None

    def test_targets_top_violation(self, sample_bgr, monkeypatch):
        from core import explainer as exp
        from core.detector import Detection, DetectionResult, Violation

        # Mock torch-only heatmap functions — return canned b64 strings
        monkeypatch.setattr(exp, "gradcam_heatmap_b64", lambda *a, **k: "FAKE_GRAD")
        monkeypatch.setattr(exp, "shap_attribution_b64", lambda *a, **k: "FAKE_SHAP")

        v1 = Violation(
            type="NO-Hardhat", risk_level="HIGH", confidence=0.91,
            bbox=(10, 10, 50, 50), person_bbox=None,
        )
        v2 = Violation(
            type="NO-Mask", risk_level="MEDIUM", confidence=0.6,
            bbox=(60, 60, 90, 90), person_bbox=None,
        )
        result = DetectionResult(
            detections=[
                Detection(cls="NO-Hardhat", confidence=0.91, bbox=(10, 10, 50, 50)),
                Detection(cls="NO-Mask", confidence=0.6, bbox=(60, 60, 90, 90)),
            ],
            violations=[v1, v2],
            image_shape=(100, 200), inference_ms=0.0,
        )
        explanation = exp.explain_result(sample_bgr, result)
        assert isinstance(explanation, Explanation)
        # Top violation = highest confidence = NO-Hardhat
        assert explanation.target_class == "NO-Hardhat"
        assert explanation.fallback_to_detection is False
        assert explanation.gradcam_b64 == "FAKE_GRAD"
        assert explanation.shap_b64 == "FAKE_SHAP"
        # annotations_b64 is the real (non-torch) function — should be a real PNG b64
        assert isinstance(explanation.annotations_b64, str)
        assert len(explanation.annotations_b64) > 0

    def test_falls_back_to_top_detection_when_no_violations(self, sample_bgr, monkeypatch):
        from core import explainer as exp
        from core.detector import Detection, DetectionResult

        monkeypatch.setattr(exp, "gradcam_heatmap_b64", lambda *a, **k: "X")
        monkeypatch.setattr(exp, "shap_attribution_b64", lambda *a, **k: "Y")

        result = DetectionResult(
            detections=[
                Detection(cls="Person", confidence=0.95, bbox=(10, 10, 50, 50)),
                Detection(cls="Hardhat", confidence=0.8, bbox=(20, 10, 40, 20)),
            ],
            violations=[],
            image_shape=(100, 200), inference_ms=0.0,
        )
        explanation = exp.explain_result(sample_bgr, result)
        assert explanation is not None
        assert explanation.target_class == "Person"  # highest-conf detection
        assert explanation.fallback_to_detection is True
