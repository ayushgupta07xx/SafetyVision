"""
SafetyVision v2 dataset merge.

Combines 4 source PPE datasets into one unified YOLO-format corpus matching
v1's 13-class schema. Performs:
  - Class remapping (source class names -> v1 canonical names, case-insensitive)
  - Perceptual-hash dedup across all datasets (catches near-duplicates)
  - Stratified 85/10/5 train/val/test split by primary class per image

Output: ./data/merged/v2/ with data.yaml + train/valid/test image+label dirs.

Usage:
    python -m model.datasets.merge

Notes:
  - v1's data.yaml is the schema source of truth. We read its class list at
    runtime so the remap dict always points at the right names.
  - If v1's class names differ from what's in REMAP below, update REMAP.
  - Perceptual-hash threshold 5 catches resize/recompression duplicates while
    keeping genuinely different scenes.
"""

import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Optional

import imagehash
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "data" / "merged" / "v2"

V1_DATA_YAML = DATA_ROOT / "v1_ppe_combined" / "data.yaml"

# Per-dataset class remaps. Keys are SOURCE class names (case-insensitive match).
# Values are TARGET v1 class names. None = DROP that class entirely.
#
# v1_ppe_combined uses PASSTHROUGH (its classes ARE the canonical schema).
REMAP: dict[str, object] = {
    "v1_ppe_combined": "PASSTHROUGH",

    "hardhat_safetyvest": {
        "helmet": "Hardhat",
        "vest": "Safety Vest",
        "head": None,  # ambiguous: could be hardhat OR no-hardhat depending on context
    },

    "fall_detection": {
        "fall-detected": "Fall-Detected",
        "Fall-Detected": "Fall-Detected",
        "fall": "Fall-Detected",
        "Fall": "Fall-Detected",
    },

    "safety_ppe_multi": {
        # Positive PPE classes
        "Helmet": "Hardhat",
        "Glove": "Gloves",
        "Goggles": "Goggles",
        "Person": "Person",
        # Violation classes
        "No_Helmet": "NO-Hardhat",
        "No_Glove": "NO-Gloves",
        "No_Goggles": "NO-Goggles",
        "No_Harness": "No_Harness",
        # Drop: not in v1 schema (preserves v1<->v2 comparability)
        "Safety_Harness": None,
        "Shoe": None,
        "No_Shoe": None,
        "No_BreathingApparatus": None,
    },

    "construction_safety_gears": {
        "Gloves": "Gloves",
        "Hardhat": "Hardhat",
        "Mask": "Mask",
        "Person": "Person",
        "Safety Vest": "Safety Vest",
        "NO-Gloves": "NO-Gloves",
        "NO-Hardhat": "NO-Hardhat",
        "NO-Mask": "NO-Mask",
        "NO-Safety Vest": "NO-Safety Vest",
        "Safety Boot": None,
        "NO-Safety Boot": None,
    },
}

PHASH_THRESHOLD = 5  # Hamming distance bits; lower = stricter dedup


def load_v1_schema() -> list[str]:
    if not V1_DATA_YAML.exists():
        raise FileNotFoundError(
            f"{V1_DATA_YAML} missing. Run `python -m model.datasets.download` first."
        )
    with open(V1_DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]
    if isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]
    log.info(f"v1 canonical schema: {len(classes)} classes -> {classes}")
    return list(classes)


def read_dataset_classes(subdir: str) -> list[str]:
    yaml_path = DATA_ROOT / subdir / "data.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]
    if isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]
    return list(classes)


def build_index_remap(
    source_classes: list[str],
    remap_dict: dict[str, Optional[str]],
    v1_classes: list[str],
) -> dict[int, Optional[int]]:
    """Build {source_idx: target_v1_idx_or_None} mapping. Case-insensitive."""
    v1_lower = {c.lower(): i for i, c in enumerate(v1_classes)}
    remap_lower = {k.lower(): v for k, v in remap_dict.items()}
    out: dict[int, Optional[int]] = {}
    for src_idx, src_name in enumerate(source_classes):
        target_name = remap_lower.get(src_name.lower())
        if target_name is None:
            out[src_idx] = None
            continue
        tgt_idx = v1_lower.get(target_name.lower())
        if tgt_idx is None:
            log.warning(f"  Target '{target_name}' not in v1 schema; dropping '{src_name}'")
            out[src_idx] = None
        else:
            out[src_idx] = tgt_idx
    return out


def remap_label_file(label_path: Path, idx_remap: dict[int, Optional[int]]) -> list[str]:
    """Read YOLO label file, remap class indices, drop labels with None target."""
    out: list[str] = []
    if not label_path.exists():
        return out
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                src_idx = int(parts[0])
            except ValueError:
                continue
            tgt = idx_remap.get(src_idx)
            if tgt is None:
                continue
            parts[0] = str(tgt)
            out.append(" ".join(parts))
    return out


def collect_pass1(
    subdir: str, idx_remap: dict[int, Optional[int]]
) -> list[tuple[Path, list[str]]]:
    """Walk dataset, return [(image_path, remapped_labels)] for images with >=1 valid label."""
    out: list[tuple[Path, list[str]]] = []
    source_root = DATA_ROOT / subdir
    for split in ("train", "valid", "test"):
        img_dir = source_root / split / "images"
        lbl_dir = source_root / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            remapped = remap_label_file(lbl_path, idx_remap)
            if not remapped:
                continue
            out.append((img_path, remapped))
    return out


def hash_image(path: Path) -> Optional[imagehash.ImageHash]:
    try:
        with Image.open(path) as im:
            return imagehash.phash(im)
    except Exception as e:
        log.warning(f"hash fail on {path.name}: {e}")
        return None


def dedup(records: list[tuple[Path, list[str], str]]) -> list[tuple[Path, list[str], str]]:
    """MD5 exact-duplicate dedup. Fast (O(N)), catches byte-identical images across datasets.
    Replaces an earlier broken pHash implementation that was effectively O(N^2)."""
    import hashlib
    log.info(f"Computing MD5 hashes for {len(records)} images...")
    seen: set[str] = set()
    kept: list[tuple[Path, list[str], str]] = []
    for img, labels, src in tqdm(records, desc="md5"):
        try:
            with open(img, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            log.warning(f"md5 fail on {img.name}: {e}")
            continue
        if md5 in seen:
            continue
        seen.add(md5)
        kept.append((img, labels, src))
    log.info(f"Kept {len(kept)} / {len(records)} ({len(records) - len(kept)} dropped as exact duplicates)")
    return kept


def primary_class(labels: list[str]) -> int:
    counts = Counter(int(line.split()[0]) for line in labels)
    return counts.most_common(1)[0][0]


def split_records(records: list, train_frac: float = 0.85, val_frac: float = 0.10):
    """Stratified split by primary class, seed=42 for reproducibility."""
    rng = Random(42)
    by_class: dict[int, list] = defaultdict(list)
    for rec in records:
        by_class[primary_class(rec[1])].append(rec)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def write_split(records: list, split_name: str) -> None:
    img_out = OUT_ROOT / split_name / "images"
    lbl_out = OUT_ROOT / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for i, (img_path, labels, src) in enumerate(tqdm(records, desc=f"write {split_name}")):
        stem = f"{src}_{i:07d}"
        dst_img = img_out / f"{stem}{img_path.suffix.lower()}"
        dst_lbl = lbl_out / f"{stem}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(labels) + "\n")


def audit_class_distribution(records: list, v1_classes: list[str], label: str) -> None:
    counter: Counter = Counter()
    for _, labels, _ in records:
        for line in labels:
            counter[int(line.split()[0])] += 1
    log.info(f"\n{label} class distribution:")
    for cls_idx in range(len(v1_classes)):
        cnt = counter.get(cls_idx, 0)
        marker = " ⚠️ UNDERREPRESENTED" if cnt < 100 else ""
        log.info(f"  [{cls_idx:2d}] {v1_classes[cls_idx]:20s}: {cnt:6d}{marker}")


def main() -> None:
    v1_classes = load_v1_schema()

    all_records: list[tuple[Path, list[str], str]] = []

    for subdir, remap_value in REMAP.items():
        log.info(f"\n=== {subdir} ===")
        src_classes = read_dataset_classes(subdir)
        log.info(f"  source classes ({len(src_classes)}): {src_classes}")

        if remap_value == "PASSTHROUGH":
            v1_lower = {c.lower(): i for i, c in enumerate(v1_classes)}
            idx_remap = {i: v1_lower.get(c.lower()) for i, c in enumerate(src_classes)}
        else:
            assert isinstance(remap_value, dict)
            idx_remap = build_index_remap(src_classes, remap_value, v1_classes)
        log.info(f"  remap (src_idx -> v1_idx): {idx_remap}")

        records = collect_pass1(subdir, idx_remap)
        log.info(f"  collected {len(records)} images with >=1 valid label")
        for img, labels in records:
            all_records.append((img, labels, subdir))

    log.info(f"\n=== Pre-dedup total: {len(all_records)} images ===")
    deduped = dedup(all_records)
    log.info(f"=== Post-dedup total: {len(deduped)} images ===")

    train, val, test = split_records(deduped)
    log.info(f"\nSplit: train={len(train)}, valid={len(val)}, test={len(test)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_split(train, "train")
    write_split(val, "valid")
    write_split(test, "test")

    data_yaml = {
        "path": str(OUT_ROOT.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(v1_classes),
        "names": v1_classes,
    }
    with open(OUT_ROOT / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    audit_class_distribution(train, v1_classes, "TRAIN")
    audit_class_distribution(val, v1_classes, "VALID")
    audit_class_distribution(test, v1_classes, "TEST")

    log.info(f"\n✅ Merged dataset written to {OUT_ROOT}")
    log.info(f"   data.yaml at {OUT_ROOT}/data.yaml")
    log.info("\nNext: confirm class distribution looks reasonable, then start training.")


if __name__ == "__main__":
    main()
