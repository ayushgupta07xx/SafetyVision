"""Tests for the violation-pairing rule.

Brief (Layer 1): "Violation requires BOTH a person AND missing PPE."

Logic lives in core.detector.PPEDetector._detect_violations. We exercise it
directly via an uninitialized PPEDetector instance (no ONNX session needed).
"""
from __future__ import annotations

import pytest

from core.detector import (
    Detection,
    PPEDetector,
    RISK_LEVELS,
    VIOLATION_CLASSES,
    Violation,
)


def _det(cls: str, bbox=(0.0, 0.0, 100.0, 100.0), conf=0.9) -> Detection:
    return Detection(cls=cls, confidence=conf, bbox=bbox)


# ─── Brief rule: violation requires BOTH a Person AND a NO-X ────────────────
class TestPersonAndMissingPPERule:
    def test_no_x_without_person_drops_silently(self, detector_instance):
        assert detector_instance._detect_violations([_det("NO-Hardhat")]) == []

    def test_person_alone_no_violation(self, detector_instance):
        assert detector_instance._detect_violations([_det("Person")]) == []

    def test_person_plus_positive_ppe_no_violation(self, detector_instance):
        dets = [_det("Person"), _det("Hardhat")]
        assert detector_instance._detect_violations(dets) == []

    def test_person_plus_overlapping_no_x_yields_violation(self, detector_instance):
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det("NO-Hardhat", bbox=(10, 0, 90, 40)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 1
        assert violations[0].type == "NO-Hardhat"
        assert violations[0].person_bbox == (0, 0, 100, 200)


# ─── IoU threshold ──────────────────────────────────────────────────────────
class TestIoUThreshold:
    def test_clear_overlap_kept(self, detector_instance):
        # NO-Hardhat fully inside Person → IoU well above PERSON_IOU_MIN (0.05)
        dets = [
            _det("Person", bbox=(0, 0, 100, 100)),
            _det("NO-Hardhat", bbox=(0, 0, 50, 50)),
        ]
        assert len(detector_instance._detect_violations(dets)) == 1

    def test_tiny_overlap_dropped(self, detector_instance):
        # 2×2 intersection / ~20000 union ≈ 0.0001 — well below 0.05 threshold
        dets = [
            _det("Person", bbox=(0, 0, 100, 100)),
            _det("NO-Hardhat", bbox=(98, 98, 198, 198)),
        ]
        assert detector_instance._detect_violations(dets) == []


# ─── Risk level mapping ─────────────────────────────────────────────────────
class TestRiskLevelAssignment:
    @pytest.mark.parametrize("cls,expected_risk", [
        ("NO-Hardhat", "HIGH"),
        ("NO-Safety Vest", "HIGH"),
        ("NO-Mask", "MEDIUM"),
    ])
    def test_risk_per_class(self, detector_instance, cls, expected_risk):
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det(cls, bbox=(10, 0, 90, 40)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert violations[0].risk_level == expected_risk

    def test_all_violation_classes_have_a_risk(self):
        for cls in VIOLATION_CLASSES:
            assert RISK_LEVELS[cls] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ─── Multi-person / multi-violation scenes ──────────────────────────────────
class TestComplexScenes:
    def test_two_persons_each_with_their_own_no_x(self, detector_instance):
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det("NO-Hardhat", bbox=(10, 0, 90, 40)),
            _det("Person", bbox=(300, 0, 400, 200)),
            _det("NO-Hardhat", bbox=(310, 0, 390, 40)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 2

    def test_one_person_two_missing_ppe(self, detector_instance):
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det("NO-Hardhat", bbox=(10, 0, 90, 40)),
            _det("NO-Safety Vest", bbox=(10, 50, 90, 150)),
        ]
        violations = detector_instance._detect_violations(dets)
        types = {v.type for v in violations}
        assert types == {"NO-Hardhat", "NO-Safety Vest"}

    def test_no_x_pairs_with_best_overlapping_person(self, detector_instance):
        # NO-Hardhat clearly overlaps Person A (huge IoU) vs Person B (zero)
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),         # A
            _det("Person", bbox=(300, 0, 400, 200)),       # B
            _det("NO-Hardhat", bbox=(20, 10, 90, 50)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 1
        assert violations[0].person_bbox == (0, 0, 100, 200)  # A wins


# ─── Violation dataclass ────────────────────────────────────────────────────
class TestViolationDataclass:
    def test_fields(self):
        v = Violation(
            type="NO-Hardhat",
            risk_level="HIGH",
            confidence=0.92,
            bbox=(0.0, 0.0, 100.0, 100.0),
            person_bbox=(0.0, 0.0, 200.0, 200.0),
        )
        assert v.type == "NO-Hardhat"
        assert v.risk_level == "HIGH"
        assert v.confidence == pytest.approx(0.92)
        assert v.person_bbox == (0.0, 0.0, 200.0, 200.0)

    def test_person_bbox_optional(self):
        v = Violation(
            type="NO-Mask", risk_level="MEDIUM", confidence=0.7,
            bbox=(0, 0, 10, 10), person_bbox=None,
        )
        assert v.person_bbox is None
