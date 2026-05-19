"""
SafetyVision v2 dataset download.

Pulls 4 PPE detection datasets from Roboflow Universe into ./data/raw/<source>/.
Reuses v1's PPE-Combined unchanged from cache if already present.

Sources:
  1. mazz-maxx/ppe-combined-9bprl-mmcaf       (v1 base, 57,904 images, 13 classes)
  2. ppe-kit-detection/hardhat-safetyvest     (~22k, head/helmet/vest)
  3. roboflow-universe-projects/fall-detection-ca3o8  (~4.5k, fall-detected)
  4. safety-jmser/safety_ppe                  (~6.6k, 12-class with harness)

Version numbers below are pinned for reproducibility. If a version 404s,
visit the dataset URL on Roboflow Universe and update the (workspace, project, version)
tuple to the latest available version.

Usage:
    python -m model.datasets.download

Env required (in .env at repo root):
    ROBOFLOW_API_KEY  (from app.roboflow.com/settings/api)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    sys.exit("ROBOFLOW_API_KEY missing — set in .env or export it")

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# (workspace, project, version, local_subdir, expected_class_count)
SOURCES = [
    ("mazz-maxx", "ppe-combined-9bprl-mmcaf", 1, "v1_ppe_combined", 13),
    ("ppe-kit-detection", "hardhat-safetyvest", 2, "hardhat_safetyvest", 3),
    ("roboflow-universe-projects", "fall-detection-ca3o8", 4, "fall_detection", 1),
    ("safety-jmser", "safety_ppe", 2, "safety_ppe_multi", 12),
]


def download_one(workspace: str, project: str, version: int, subdir: str) -> Path:
    target = DATA_ROOT / subdir
    if (target / "data.yaml").exists():
        print(f"[{subdir}] already downloaded, skipping (delete folder to re-pull)")
        return target

    print(f"[{subdir}] downloading {workspace}/{project} v{version}...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    proj = rf.workspace(workspace).project(project)
    proj.version(version).download("yolov8", location=str(target))
    print(f"[{subdir}] done -> {target}")
    return target


def count_split(subdir_path: Path, split: str) -> int:
    d = subdir_path / split / "images"
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main() -> None:
    print(f"Downloading 4 datasets into {DATA_ROOT}\n")
    for w, p, v, s, _ in SOURCES:
        download_one(w, p, v, s)

    print("\n=== Download summary ===")
    total = 0
    for _, _, _, s, _ in SOURCES:
        sub = DATA_ROOT / s
        n_train = count_split(sub, "train")
        n_val = count_split(sub, "valid")
        n_test = count_split(sub, "test")
        n_total = n_train + n_val + n_test
        total += n_total
        print(f"  {s}: train={n_train}, val={n_val}, test={n_test} (total {n_total})")
    print(f"\nGrand total (raw, pre-merge): {total} images")
    print(f"\nNext: python -m model.datasets.merge")


if __name__ == "__main__":
    main()
