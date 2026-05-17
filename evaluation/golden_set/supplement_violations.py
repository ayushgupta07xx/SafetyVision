"""Supplement test_split with 10 violation-rich images.

Scans the Kaggle zip's test labels for files containing NO-X / No_Harness /
Fall-Detected class ids, samples 10 (deterministic, seed=137 — different from
the original seed=42 to avoid overlap), excludes anything already in the
existing test_sample.txt, and extracts to img_test_11..20.

Appends paths to /tmp/test_sample.txt so build_ground_truth picks up all 20.

Usage:
    python -m evaluation.golden_set.supplement_violations
"""
from __future__ import annotations

import random
import shutil
import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path("/mnt/e/_output_.zip")
SAMPLE_TXT = Path("/tmp/test_sample.txt")
IMAGES_OUT = Path("evaluation/golden_set/images/test_split")
LABELS_OUT = Path("evaluation/golden_set/labels")

# Class ids that mark a violation (per data.yaml):
#   0=Fall-Detected, 5=NO-Gloves, 6=NO-Goggles, 7=NO-Hardhat,
#   8=NO-Mask, 9=NO-Safety Vest, 10=No_Harness
VIOLATION_CLASS_IDS = {0, 5, 6, 7, 8, 9, 10}

N_NEW = 10
SEED = 137


def label_has_violation(text: str) -> bool:
    """Return True if any YOLO label line starts with a violation class id."""
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        if cid in VIOLATION_CLASS_IDS:
            return True
    return False


def main() -> None:
    if not ZIP_PATH.exists():
        sys.exit(f"Zip not found: {ZIP_PATH}")
    if not SAMPLE_TXT.exists():
        sys.exit(f"{SAMPLE_TXT} missing — run extract_yolo_labels first")

    existing_paths = {line.strip() for line in SAMPLE_TXT.read_text().splitlines() if line.strip()}
    print(f"Existing test_split images: {len(existing_paths)}")

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.mkdir(parents=True, exist_ok=True)

    print("Scanning zip for violation-bearing labels (this takes ~30s)...")
    candidates: list[tuple[str, str]] = []  # (image_path, label_path)

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        label_names = [n for n in names if n.startswith("PPE-Combined-1/test/labels/")
                       and n.endswith(".txt")]
        print(f"  {len(label_names)} test label files to scan")

        for label_name in label_names:
            with z.open(label_name) as f:
                text = f.read().decode("utf-8", errors="ignore")
            if not label_has_violation(text):
                continue
            # Map label → image
            image_name = (
                label_name.replace("/labels/", "/images/").rsplit(".txt", 1)[0] + ".jpg"
            )
            if image_name in existing_paths:
                continue
            if image_name not in names:
                continue  # image missing for some reason
            candidates.append((image_name, label_name))

    print(f"  {len(candidates)} violation-bearing images available (excluding existing 10)")

    if len(candidates) < N_NEW:
        sys.exit(f"Not enough candidates ({len(candidates)} < {N_NEW})")

    random.seed(SEED)
    sampled = random.sample(candidates, N_NEW)
    sampled.sort()  # deterministic ordering for case naming

    print(f"\nExtracting {N_NEW} new images + labels...")
    new_image_paths = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        for offset, (img_path, lab_path) in enumerate(sampled, start=11):
            img_out = IMAGES_OUT / f"img_test_{offset:02d}.jpg"
            lab_out = LABELS_OUT / f"img_test_{offset:02d}.txt"
            with z.open(img_path) as src, open(img_out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            with z.open(lab_path) as src, open(lab_out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            new_image_paths.append(img_path)
            print(f"  img_test_{offset:02d}: {img_path.rsplit('/', 1)[-1]}")

    # Append to test_sample.txt so build_ground_truth picks them up
    with open(SAMPLE_TXT, "a") as f:
        for p in new_image_paths:
            f.write(p + "\n")

    print(f"\nAppended {N_NEW} paths to {SAMPLE_TXT}")
    print(f"Total test_split now: {len(existing_paths) + N_NEW}")
    print("\nNext: python -m evaluation.golden_set.build_ground_truth")


if __name__ == "__main__":
    main()
