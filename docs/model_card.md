---
license: agpl-3.0
language:
- en
library_name: ultralytics
pipeline_tag: object-detection
tags:
- yolov8
- object-detection
- ppe-detection
- workplace-safety
- computer-vision
- onnx
- safety
- osha
---

# SafetyVision YOLOv8n — PPE Detection

YOLOv8n fine-tuned for Personal Protective Equipment (PPE) detection at industrial worksites. Backbone model for [SafetyVision](https://github.com/ayushgupta07xx/SafetyVision), an open-source AI workplace safety monitor.

| Headline metric | Value |
|---|---|
| **Test mAP@0.5** | **0.701** |
| **Test mAP@0.5:0.95** | **0.441** |
| Parameters | 3,008,183 (~3M) |
| FLOPs | 8.1 GFLOPs |
| Input | 640×640 RGB |
| Inference (T4 GPU) | 2.9 ms |

## Model description

YOLOv8n fine-tuned for 13-class PPE detection covering hard hats, safety vests, goggles, gloves, masks, fall harnesses, their "missing/no" violation counterparts, fall detection, and a Person class.

- **Base model:** [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- **Output:** 17 channels × 8400 anchor predictions → NMS → bounding boxes + class labels + confidence scores
- **Use it for:** flagging likely PPE violations in static images and short video clips for human review
- **Do not use it for:** automated disciplinary action, medical/clinical PPE, food safety, hazmat suits, or any standalone enforcement decision

## Classes

| ID | Class | Type |
|----|---|---|
| 0 | Fall-Detected | Event |
| 1 | Gloves | PPE worn ✓ |
| 2 | Goggles | PPE worn ✓ |
| 3 | Hardhat | PPE worn ✓ |
| 4 | Mask | PPE worn ✓ |
| 5 | NO-Gloves | Violation ✗ |
| 6 | NO-Goggles | Violation ✗ |
| 7 | NO-Hardhat | Violation ✗ |
| 8 | NO-Mask | Violation ✗ |
| 9 | NO-Safety Vest | Violation ✗ |
| 10 | No_Harness | Violation ✗ |
| 11 | Person | Person detection |
| 12 | Safety Vest | PPE worn ✓ |

## Training data

[PPE-Combined v1](https://universe.roboflow.com/mazz-maxx/ppe-combined-9bprl-mmcaf) (Roboflow Universe, forked from `s-workspace-cjeuu/ppe-combined-9bprl`).

| Split | Images | Annotated instances |
|---|---|---|
| Train | 41,922 | — |
| Validation | 10,834 | 25,308 |
| Test (held-out) | 5,148 | 11,303 |
| **Total** | **57,904** | |

## Training procedure

- **Hardware:** Kaggle Notebooks, 2× NVIDIA Tesla T4 (15.6 GB VRAM each)
- **Framework:** Ultralytics 8.3.40, PyTorch 2.10.0 + CUDA 12.8
- **Epochs:** 100
- **Batch size:** 32
- **Image size:** 640×640
- **Optimizer:** SGD (lr0=0.01, momentum=0.937, weight_decay=0.0005, default ultralytics schedule)
- **Wall time:** ~15 hours across two Kaggle Save Versions (12-hour session cap forced a mid-run resume at epoch 82)
- **Resume:** Save Version 1 crashed at epoch 82.5/100 on the Kaggle 12-hour cap. Save Version 2 resumed from the `epoch82.pt` checkpoint via `wandb.init(id="9nctv2ai", resume="must", settings=wandb.Settings(init_timeout=300))` and completed epochs 83–100 cleanly.

### Experiment tracking

- **W&B run** (public): https://wandb.ai/agcr7jw-vellore-institute-of-technology/safetyvision/runs/9nctv2ai
  - Epochs 1–82: full charts available
  - Epochs 83–100: W&B callback didn't fire after resume (known ultralytics+wandb sharp edge). Canonical metrics for all 100 epochs are in [`model/yolov8n-ppe-v1/results.csv`](https://github.com/ayushgupta07xx/SafetyVision/blob/main/model/yolov8n-ppe-v1/results.csv) in the repo.
- **MLflow:** local file-based tracking committed at [`mlruns/`](https://github.com/ayushgupta07xx/SafetyVision/tree/main/mlruns), run ID `f1932e539038417dad6db757affd50e6`.

## Evaluation

### Headline metrics (held-out test split, 5,148 images)

| Metric | Test | Validation |
|---|---|---|
| **mAP@0.5** | **0.701** | 0.693 |
| **mAP@0.5:0.95** | **0.441** | 0.431 |
| Precision | 0.607 | 0.629 |
| Recall | 0.711 | 0.716 |

Test split is slightly *better* than validation — clean generalization signal, no overfitting.

### Per-class test metrics

| Class | Instances | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Fall-Detected | 450 | 0.775 | 0.793 | 0.838 | 0.555 |
| Gloves | 569 | 0.765 | 0.715 | 0.802 | 0.416 |
| **Goggles** | 470 | 0.798 | 0.860 | **0.908** | 0.528 |
| **Hardhat** | 5,100 | 0.813 | 0.883 | **0.891** | 0.518 |
| Mask | 434 | 0.568 | 0.733 | 0.651 | 0.394 |
| NO-Gloves | 676 | 0.740 | 0.754 | 0.802 | 0.400 |
| NO-Goggles | 558 | 0.780 | 0.701 | 0.799 | 0.458 |
| NO-Hardhat | 1,122 | 0.621 | 0.821 | 0.775 | 0.521 |
| NO-Mask | 218 | 0.502 | 0.763 | 0.573 | 0.378 |
| NO-Safety Vest | 342 | 0.469 | 0.491 | 0.395 | 0.217 |
| No_Harness | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| Person | 138 | 0.367 | 0.942 | 0.864 | 0.739 |
| Safety Vest | 1,225 | 0.692 | 0.789 | 0.815 | 0.615 |

Curves and confusion matrices for both splits are committed in [`model/yolov8n-ppe-v1/`](https://github.com/ayushgupta07xx/SafetyVision/tree/main/model/yolov8n-ppe-v1).

## Inference performance

- **GPU (Tesla T4):** preprocess 0.2 ms + inference 2.9 ms + postprocess 1.0 ms = **~4 ms/image total**

## Intended use

Pre-screening tool to **assist** human workplace safety officers by surfacing likely PPE violations in images and short video clips for human review. Designed for:

- Construction sites
- Warehouses
- Manufacturing floors
- Pre-shift safety walkthroughs

**Not a replacement for human judgment.** Predictions must be reviewed by qualified safety personnel before any disciplinary, compliance, or insurance action.

## Out of scope

- Medical/clinical settings (different PPE: gowns, N95 fit testing, sterile gloves)
- Food processing (hairnets, beard guards, lab coats not represented)
- Chemical/hazmat operations (full-face respirators, encapsulating suits not represented)
- Drone or overhead camera angles (training data is ground-level/eye-level)
- Crowded scenes with heavy mutual occlusion
- Real-time alerting where missing a violation is unacceptable

## Failure modes

Documented from training data review and observed test errors:

- **Low light or high glare** — Confidence drops sharply; expect both false positives and false negatives.
- **Partial occlusion** — Workers partially behind machinery or other workers may have PPE missed.
- **Unusual PPE colors** — Training data skews to high-vis yellow/orange/lime vests and standard white/yellow hard hats. Rare colors (blue, black) may go undetected.
- **Small workers (<50 px height)** — Distant figures often missed entirely.
- **Fast motion in video** — Motion blur causes missed frames. Mitigate by aggregating across multiple frames per scene rather than relying on any single frame.
- **`No_Harness` class** — Severely underrepresented in training and test data (1 test instance, 276 val instances). Effectively unusable as a detector until training data is augmented. **Do not rely on this class for fall-arrest compliance.**
- **`NO-Safety Vest` class** — Weakest violation class (test mAP 0.395). High false-negative rate.

## Bias and limitations

- Training data over-represents Western construction/industrial sites; demographics and PPE conventions in South/Southeast Asia, Africa, and the Middle East may be underrepresented.
- Heavily skewed toward male-presenting workers in training imagery.
- The Person class inherits biases from the underlying YOLOv8 COCO pretraining.
- Indoor warehouse lighting is overrepresented; bright outdoor sun and underground/tunnel environments may degrade performance.
- 13 classes is a fixed taxonomy — site-specific PPE (e.g., arc-flash hoods, cut-resistant sleeves) will not be detected.

## Files

| File | Size | Description |
|---|---|---|
| `best.pt` | 6.0 MB | PyTorch weights, load with `ultralytics.YOLO("best.pt")` |
| `best.onnx` | 11.6 MB | ONNX export (opset 18, slimmed with onnxslim 0.1.93) |
| `best.onnx.data` | 11.6 MB | External weights for `best.onnx` — **must be co-located** with the .onnx file |

## Usage

### PyTorch (ultralytics)

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

weights = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="best.pt")
model = YOLO(weights)
results = model("worksite_image.jpg")
results[0].show()
```

### ONNX Runtime (CPU-friendly, used in AWS Lambda)

```python
import cv2, numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# Both .onnx and .onnx.data must be in the same directory
hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="best.onnx.data")
onnx_path = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="best.onnx")

session = ort.InferenceSession(onnx_path)
img = cv2.imread("worksite_image.jpg")
img = cv2.resize(img, (640, 640))
inp = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
outputs = session.run(None, {"images": inp})
# outputs[0] shape: (1, 17, 8400) — apply your own NMS for final boxes
```

## License

- **Model weights:** AGPL-3.0 (inherited from Ultralytics YOLOv8 base model)
- **SafetyVision repository code:** see [LICENSE](https://github.com/ayushgupta07xx/SafetyVision/blob/main/LICENSE) in the project repo.

## Citation

```bibtex
@software{safetyvision_2026,
  author = {Gupta, Ayush},
  title  = {SafetyVision: Open-Source AI Workplace Safety Monitor},
  year   = {2026},
  url    = {https://github.com/ayushgupta07xx/SafetyVision}
}
```

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 and the training framework
- [Roboflow Universe](https://universe.roboflow.com) and the PPE-Combined dataset maintainers
- [OSHA](https://www.osha.gov) for the public-domain regulation corpus used in incident report generation
- [Kaggle Notebooks](https://www.kaggle.com/code) for free 2× T4 GPU training
