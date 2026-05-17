"""Extract YOLO labels + data.yaml from the Kaggle zip for our 10 test_split images.

The PPE-Combined-1 dataset stores labels at:
    PPE-Combined-1/test/labels/<same-basename>.txt
in standard YOLO format (one line per object):
    <class_id> <x_center> <y_center> <width> <height>     # all normalized [0,1]

This script pulls:
  1. data.yaml (the class_id → name mapping)
  2. Label .txt files for the 10 images listed in /tmp/test_sample.txt

Output:
    evaluation/golden_set/data.yaml
    evaluation/golden_set/labels/img_test_NN.txt   (renamed to match our images)

Usage:
    python -m evaluation.golden_set.extract_yolo_labels
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path("/mnt/e/_output_.zip")
SAMPLE_TXT = Path("/tmp/test_sample.txt")
OUT_DIR = Path("evaluation/golden_set")
LABELS_DIR = OUT_DIR / "labels"


def main() -> None:
    if not ZIP_PATH.exists():
        sys.exit(f"Zip not found at {ZIP_PATH}")
    if not SAMPLE_TXT.exists():
        sys.exit(
            f"{SAMPLE_TXT} not found — this is the list of original image paths "
            "(from the test_split extraction step). Re-run the test_split sampling."
        )

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = [line.strip() for line in SAMPLE_TXT.read_text().splitlines() if line.strip()]
    # Derive label paths: swap /images/ → /labels/ and .jpg → .txt
    label_paths = [
        p.replace("/images/", "/labels/").rsplit(".jpg", 1)[0] + ".txt"
        for p in image_paths
    ]

    with zipfile.ZipFile(ZIP_PATH) as z:
        # 1. Pull data.yaml — try common locations
        yaml_path = None
        for candidate in ["PPE-Combined-1/data.yaml", "data.yaml"]:
            try:
                with z.open(candidate) as src, open(OUT_DIR / "data.yaml", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                yaml_path = candidate
                break
            except KeyError:
                continue
        if not yaml_path:
            print("WARNING: no data.yaml found in zip — will need manual class map")
        else:
            print(f"Extracted data.yaml from {yaml_path}")

        # 2. Pull the 10 label files, renaming to img_test_NN.txt to match our images
        all_in_zip = set(z.namelist())
        extracted = 0
        for i, label_path in enumerate(label_paths, 1):
            target = LABELS_DIR / f"img_test_{i:02d}.txt"
            if label_path not in all_in_zip:
                # Some images may have empty label files (negative examples)
                # which means YOLO labelers omitted them. Write empty file.
                target.write_text("")
                print(f"  img_test_{i:02d}: no label in zip (treated as empty) — {label_path}")
                continue
            with z.open(label_path) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

        print(f"Extracted {extracted}/{len(label_paths)} label files to {LABELS_DIR}")

    # Sanity: print summary of each label
    print("\nLabel summary:")
    for label_file in sorted(LABELS_DIR.glob("*.txt")):
        text = label_file.read_text().strip()
        n_objects = len(text.splitlines()) if text else 0
        print(f"  {label_file.name}: {n_objects} objects")


if __name__ == "__main__":
    main()
