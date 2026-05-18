"""Tests for the violation surfacing rule.

ADR-010: every NO-X detection surfaces as a violation. When a Person bbox
pairs (IoU >= PERSON_IOU_MIN), it is attached to the violation for richer
downstream reporting; otherwise person_bbox is None.

Logic lives in core.detector.PPEDetector._detect_violations. We exercise it
directly via an uninitialized PPEDetector instance (no ONNX session needed).
"""
from __future__ import annotations

import pytest

from core.detector import (
    RISK_LEVELS,
    VIOLATION_CLASSES,
    Detection,
    Violation,
)


def _det(cls: str, bbox=(0.0, 0.0, 100.0, 100.0), conf=0.9) -> Detection:
    return Detection(cls=cls, confidence=conf, bbox=bbox)


# ─── Raw NO-X surfacing (ADR-010) ───────────────────────────────────────────
class TestRawNoXSurfacing:
    def test_no_x_without_person_surfaces_unpaired(self, detector_instance):
        """ADR-010: NO-X alone (e.g. occluded forklift driver) is a violation."""
        violations = detector_instance._detect_violations([_det("NO-Hardhat")])
        assert len(violations) == 1
        assert violations[0].type == "NO-Hardhat"
        assert violations[0].person_bbox is None

    def test_person_alone_no_violation(self, detector_instance):
        assert detector_instance._detect_violations([_det("Person")]) == []

    def test_person_plus_positive_ppe_no_violation(self, detector_instance):
        dets = [_det("Person"), _det("Hardhat")]
        assert detector_instance._detect_violations(dets) == []

    def test_person_plus_overlapping_no_x_yields_paired_violation(self, detector_instance):
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det("NO-Hardhat", bbox=(10, 0, 90, 40)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 1
        assert violations[0].type == "NO-Hardhat"
        assert violations[0].person_bbox == (0, 0, 100, 200)


# ─── IoU-based Person pairing (when Person bbox attaches) ───────────────────
class TestIoUPairing:
    def test_clear_overlap_attaches_person(self, detector_instance):
        dets = [
            _det("Person", bbox=(0, 0, 100, 100)),
            _det("NO-Hardhat", bbox=(0, 0, 50, 50)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 1
        assert violations[0].person_bbox == (0, 0, 100, 100)

    def test_tiny_overlap_surfaces_violation_without_person(self, detector_instance):
        """Person present but pairing IoU below threshold → violation surfaces
        anyway with person_bbox=None (ADR-010)."""
        dets = [
            _det("Person", bbox=(0, 0, 100, 100)),
            _det("NO-Hardhat", bbox=(98, 98, 198, 198)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 1
        assert violations[0].type == "NO-Hardhat"
        assert violations[0].person_bbox is None


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
        assert all(v.person_bbox is not None for v in violations)

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
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det("Person", bbox=(300, 0, 400, 200)),
            _det("NO-Hardhat", bbox=(20, 10, 90, 50)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 1
        assert violations[0].person_bbox == (0, 0, 100, 200)

    def test_mixed_paired_and_unpaired_violations_in_same_scene(self, detector_instance):
        """Real-world scene: one visible worker, one occluded worker."""
        dets = [
            _det("Person", bbox=(0, 0, 100, 200)),
            _det("NO-Hardhat", bbox=(10, 0, 90, 40)),
            _det("NO-Hardhat", bbox=(500, 100, 580, 160)),
        ]
        violations = detector_instance._detect_violations(dets)
        assert len(violations) == 2
        paired = [v for v in violations if v.person_bbox is not None]
        unpaired = [v for v in violations if v.person_bbox is None]
        assert len(paired) == 1
        assert len(unpaired) == 1


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
