"""
Phase 2 v2 training — YOLOv8s on merged ~80k PPE dataset.

Run modes:
  Dry run (~5 min, 1% data, verifies pipeline):
    python -m model.train_v2 --dry-run

  Full training (~57 hrs on L4):
    python -m model.train_v2

Auto-resumes from last.pt if a prior run was interrupted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load env BEFORE importing ultralytics so WANDB_API_KEY is visible
load_dotenv()


def patch_albumentations() -> None:
    """Replace ultralytics' default Albumentations transforms with our custom pipeline.

    Calls the original __init__ first to let ultralytics set up internal state,
    then overrides self.transform with our list. Robust across ultralytics versions.
    """
    import albumentations as A
    from ultralytics.data.augment import Albumentations

    _original_init = Albumentations.__init__

    def custom_init(self, p: float = 1.0) -> None:
        _original_init(self, p=p)
        T = [
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.5,
            ),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ]
        self.transform = A.Compose(
            T,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        print("[train_v2] Albumentations patched: CoarseDropout + MotionBlur + RandomGamma + CLAHE + Perspective")

    Albumentations.__init__ = custom_init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="1 epoch on 1% of data, multi_scale off")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    mlruns_dir = repo_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)

    # MLflow: local file-based, committed to repo
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "safetyvision-v2")

    # W&B: project pinned, run name set via ultralytics 'name' arg
    os.environ.setdefault("WANDB_PROJECT", "safetyvision")

    # Enable ultralytics' built-in W&B + MLflow callbacks
    from ultralytics import settings
    settings.update({"wandb": True, "mlflow": True})

    # Monkey-patch BEFORE importing YOLO so the patched class is used
    patch_albumentations()

    from ultralytics import YOLO

    name = "yolov8s-ppe-v2-dryrun" if args.dry_run else "yolov8s-ppe-v2"
    data_yaml = repo_root / "data" / "merged" / "v2" / "data.yaml"
    last_pt = repo_root / "runs" / "detect" / name / "weights" / "last.pt"

    if last_pt.exists():
        print(f"[train_v2] Resuming from {last_pt}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        print("[train_v2] Starting fresh from yolov8s.pt")
        model = YOLO("yolov8s.pt")
        resume = False

    # Locked config from chat-9 handoff
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=1 if args.dry_run else 150,
        imgsz=896,
        batch=24,
        device=0,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        patience=25,
        close_mosaic=15,
        save_period=5,
        multi_scale=not args.dry_run,  # off in dry-run; on in production
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fraction=0.01 if args.dry_run else 1.0,
        name=name,
        exist_ok=True,
        resume=resume,
        verbose=True,
    )

    print(f"[train_v2] Launching: name={name} epochs={train_kwargs['epochs']} "
          f"imgsz={train_kwargs['imgsz']} batch={train_kwargs['batch']} "
          f"multi_scale={train_kwargs['multi_scale']} fraction={train_kwargs['fraction']}")
    model.train(**train_kwargs)
    print("[train_v2] Training complete")


if __name__ == "__main__":
    main()
