"""Build the canonical golden-set ground truth.

Combines:
  - test_split: YOLO labels from the dataset (objective, from data.yaml class map)
  - unsplash:   manual annotations in unsplash_ground_truth.json

Applies the deterministic regulation_map (class + context → expected CFR + risk)
to produce the final cases.json. NO LLM is involved in ground truth — every
expected value is either from the dataset or from a curated map.

Output schema (cases.json):
    {
      "img_test_01": {
        "image_path": "...",
        "source": "test_split",
        "original_filename": "...",
        "ground_truth": {
          "context": "construction",
          "visible_classes": ["Person", "Mask"],
          "violations": [
            {
              "type": "NO-Hardhat",
              "expected_regulation": "29 CFR 1926.100",
              "expected_risk_level": "HIGH"
            }
          ]
        }
      }
    }

If a case has empty `violations`, the scene is compliant. The A/B test #1
filters to cases with violations only (compliant cases are reported separately
as a false-positive sanity check).

Usage:
    python -m evaluation.golden_set.build_ground_truth
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from evaluation.golden_set.regulation_map import (
    VIOLATION_CLASSES,
    expected_regulation,
    expected_risk,
)

GOLDEN_DIR = Path("evaluation/golden_set")
IMAGES_DIR = GOLDEN_DIR / "images"
LABELS_DIR = GOLDEN_DIR / "labels"
DATA_YAML = GOLDEN_DIR / "data.yaml"
UNSPLASH_GT = GOLDEN_DIR / "unsplash_ground_truth.json"
SAMPLE_TXT = Path("/tmp/test_sample.txt")
OUT_PATH = GOLDEN_DIR / "cases.json"

# Default context for test_split images. PPE-Combined dataset spans both
# construction and general-industry scenes, but without per-image context
# labels the conservative default is 'construction' (broader CFR Part 1926
# coverage matches the common case for hardhat / vest / harness violations).
TEST_SPLIT_DEFAULT_CONTEXT = "construction"


def load_class_names() -> dict[int, str]:
    """Parse data.yaml → {class_id: class_name}."""
    if not DATA_YAML.exists():
        sys.exit(
            f"{DATA_YAML} not found. Run `python -m evaluation.golden_set.extract_yolo_labels` first."
        )
    data = yaml.safe_load(DATA_YAML.read_text())
    names = data.get("names")
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    sys.exit(f"Unexpected names format in {DATA_YAML}: {type(names)}")


def parse_yolo_label(path: Path, class_names: dict[int, str]) -> list[str]:
    """Return the list of class names present in a YOLO label file.
    Returns [] for empty/missing files (negative examples, fully compliant scenes).
    """
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    classes = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid in class_names:
            classes.append(class_names[cid])
    return classes


def build_test_split_cases(class_names: dict[int, str]) -> dict:
    """Build cases for the 10 test_split images from YOLO labels."""
    if not SAMPLE_TXT.exists():
        sys.exit(f"{SAMPLE_TXT} not found — original test image paths needed for traceability.")
    original_paths = [line.strip() for line in SAMPLE_TXT.read_text().splitlines() if line.strip()]

    cases = {}
    for i, orig in enumerate(original_paths, 1):
        case_id = f"img_test_{i:02d}"
        image_path = IMAGES_DIR / "test_split" / f"{case_id}.jpg"
        label_path = LABELS_DIR / f"{case_id}.txt"

        visible_classes = parse_yolo_label(label_path, class_names)
        # Deduplicate while preserving order
        seen = set()
        visible_unique = [c for c in visible_classes if not (c in seen or seen.add(c))]

        # Violations = visible NO-X / No_Harness / Fall-Detected classes
        violation_types = [c for c in visible_unique if c in VIOLATION_CLASSES]

        violations = [
            {
                "type": v,
                "expected_regulation": expected_regulation(v, TEST_SPLIT_DEFAULT_CONTEXT),
                "expected_risk_level": expected_risk(v),
            }
            for v in violation_types
        ]

        cases[case_id] = {
            "image_path": str(image_path),
            "source": "test_split",
            "original_filename": orig.rsplit("/", 1)[-1],
            "ground_truth": {
                "context": TEST_SPLIT_DEFAULT_CONTEXT,
                "visible_classes": visible_unique,
                "violations": violations,
            },
        }
    return cases


def build_unsplash_cases() -> dict:
    """Build cases for the 10 Unsplash images from manual annotation."""
    if not UNSPLASH_GT.exists():
        sys.exit(f"{UNSPLASH_GT} not found.")
    raw = json.loads(UNSPLASH_GT.read_text())

    if not raw.get("_filled"):
        print(
            "WARNING: unsplash_ground_truth.json has _filled=false — "
            "you haven't reviewed it yet. Continuing with template values "
            "(which I pre-filled from filenames). Review and set _filled=true before "
            "running the A/B test."
        )

    cases = {}
    for case_id, gt in raw.items():
        if case_id.startswith("_"):
            continue  # skip _README, _filled

        image_path = IMAGES_DIR / "unsplash" / f"{case_id}.jpg"
        if not image_path.exists():
            print(f"  WARN: image not found for {case_id} at {image_path}")
            continue

        context = gt.get("context", "construction")
        visible = gt.get("visible_classes", [])
        violation_types = gt.get("violations", [])

        violations = [
            {
                "type": v,
                "expected_regulation": expected_regulation(v, context),
                "expected_risk_level": expected_risk(v),
            }
            for v in violation_types
        ]

        cases[case_id] = {
            "image_path": str(image_path),
            "source": "unsplash",
            "ground_truth": {
                "context": context,
                "visible_classes": visible,
                "violations": violations,
            },
        }
    return cases


def main() -> None:
    class_names = load_class_names()
    print(f"Loaded {len(class_names)} class names from data.yaml:")
    for cid, name in sorted(class_names.items()):
        print(f"  {cid}: {name}")

    test_cases = build_test_split_cases(class_names)
    unsplash_cases = build_unsplash_cases()

    cases = {**test_cases, **unsplash_cases}

    # Back up the previous (model-derived) cases.json
    if OUT_PATH.exists():
        backup = OUT_PATH.with_suffix(".json.preexpertpath.bak")
        OUT_PATH.rename(backup)
        print(f"\nBacked up previous cases.json -> {backup}")

    OUT_PATH.write_text(json.dumps(cases, indent=2))

    n_total = len(cases)
    n_violations = sum(1 for c in cases.values() if c["ground_truth"]["violations"])
    n_compliant = n_total - n_violations
    types_seen = sorted({
        v["type"]
        for c in cases.values()
        for v in c["ground_truth"]["violations"]
    })

    print(f"\nWrote {OUT_PATH}")
    print(f"  Total cases:    {n_total}")
    print(f"  With violations: {n_violations}  (used in A/B test #1)")
    print(f"  Compliant:       {n_compliant}  (tracked separately for false-positive check)")
    print(f"  Violation types covered: {types_seen}")


if __name__ == "__main__":
    main()
