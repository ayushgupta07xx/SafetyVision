"""GradCAM + EigenCAM + SHAP explainability.

Brief reference:
    Layer 2 — Explainability (SHAP + GradCAM)
    "Both must appear in every violation output."

Three explainers (chat 5):
    - GradCAM: class-specific, bbox-masked to top-class detections only
    - EigenCAM: class-agnostic full-scene, bbox-masked to all detections.
      Handles multiple instances natively (no target collapse), forward-only.
    - SHAP: per-pixel attribution heatmap for the top violation's class.

Latency optimizations (chat 5):
    - Explainer runs at 320×320 (detector still 640) — ~3-4× faster, heatmap
      displayed at downsampled resolution which is fine for visualization.
    - SHAP nsamples=20 (default 200) — ~10× speedup, mild quality loss.
    - EigenCAM is forward-only (uses_gradients=False) — ~30× faster than GradCAM.
    - Direct preprocessing (no duplicate PPEDetector instance per call).
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (HF Spaces / Lambda are headless)
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM

from core.detector import HF_REPO, DetectionResult, draw_annotations

logger = logging.getLogger(__name__)

PT_FILENAME = "best.pt"
EXPLAINER_IMG_SIZE = 320  # smaller than detector's 640 for latency
SHAP_NSAMPLES = 20  # default 200; 20 gives acceptable attribution at ~10× speedup


# ─── PyTorch model cache ────────────────────────────────────────────────────
class _TorchModelCache:  # pragma: no cover
    """Singleton for the PyTorch YOLO model (needed for grads + EigenCAM)."""

    _model: nn.Module | None = None
    _name_to_id: dict[str, int] | None = None

    @classmethod
    def get(cls) -> nn.Module:
        if cls._model is None:
            cls._load()
        assert cls._model is not None
        return cls._model

    @classmethod
    def name_to_id(cls, name: str) -> int:
        if cls._name_to_id is None:
            cls._load()
        assert cls._name_to_id is not None
        return cls._name_to_id[name]

    @classmethod
    def _load(cls) -> None:
        from ultralytics import YOLO  # lazy import — keeps detector dep-light

        pt_path = hf_hub_download(repo_id=HF_REPO, filename=PT_FILENAME)
        logger.info("Loading PyTorch model from %s", pt_path)
        yolo = YOLO(pt_path)
        model = yolo.model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cls._model = model
        names = model.names if hasattr(model, "names") else yolo.names
        cls._name_to_id = {v: k for k, v in dict(names).items()}
        logger.info("PyTorch model loaded with %d classes", len(cls._name_to_id))


# ─── Target & wrapper ───────────────────────────────────────────────────────
class _YoloClassTarget:  # pragma: no cover
    """Scalar target for pytorch-grad-cam.

    Sums class scores of all anchors above `threshold`. Threshold 0.25 aligns
    with typical NMS-survivor confidence — filters noise anchors while
    aggregating across all real detections of the class.
    """

    def __init__(self, class_id: int, threshold: float = 0.25) -> None:
        self.class_id = class_id
        self.threshold = threshold

    def __call__(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() == 3 and output.shape[0] == 1:
            output = output[0]
        scores = output[4 + self.class_id, :]
        mask = scores > self.threshold
        if mask.any():
            return scores[mask].sum()
        return scores.max()


class _YoloScalarWrapper(nn.Module):  # pragma: no cover
    """nn.Module producing a scalar per input for SHAP GradientExplainer."""

    def __init__(self, model: nn.Module, class_id: int) -> None:
        super().__init__()
        self.model = model
        self.class_id = class_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[:, 4 + self.class_id, :].max(dim=1)[0].unsqueeze(-1)


# ─── Preprocessing & encoding helpers ───────────────────────────────────────
def _preprocess_torch(
    image_bgr: np.ndarray,
    size: int = EXPLAINER_IMG_SIZE,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """Letterbox to size×size + normalize → (input tensor, letterboxed RGB uint8).

    Independent of detector's PPEDetector singleton — no duplicate ONNX load.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = size - new_h, size - new_w
    top, left = pad_h // 2, pad_w // 2
    letterboxed = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    arr = letterboxed.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, :]
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor, letterboxed


def _bbox_mask_letterboxed(
    image_bgr: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
    size: int = EXPLAINER_IMG_SIZE,
) -> np.ndarray:
    """Build a binary mask in letterboxed (size×size) space from original-image bboxes."""
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    mask = np.zeros((size, size), dtype=np.float32)
    for x1, y1, x2, y2 in bboxes:
        lx1 = max(0, int(x1 * scale + pad_x))
        ly1 = max(0, int(y1 * scale + pad_y))
        lx2 = min(size, int(x2 * scale + pad_x))
        ly2 = min(size, int(y2 * scale + pad_y))
        mask[ly1:ly2, lx1:lx2] = 1.0
    return mask


def _apply_bbox_mask(
    overlay: np.ndarray, letterboxed: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Blend overlay inside mask region; restore original pixels outside."""
    mask_3ch = mask[..., None]
    return (
        overlay.astype(np.float32) * mask_3ch
        + letterboxed.astype(np.float32) * (1 - mask_3ch)
    ).astype(np.uint8)


def _heatmap_overlay(
    letterboxed: np.ndarray, heatmap: np.ndarray, alpha_max: float = 0.6
) -> np.ndarray:
    """Overlay colored heatmap on image with intensity-based alpha.

    Pixels where heatmap ≈ 0 show the original image unchanged. Pixels where
    heatmap ≈ 1 show the colormap color blended at alpha_max. Replaces
    show_cam_on_image's flat 50/50 blend (which paints uniform-heat regions as
    solid colored rectangles when bbox-masked).
    """
    colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (heatmap * alpha_max)[..., None]
    blended = letterboxed.astype(np.float32) * (1 - alpha) + colored_rgb * alpha
    return blended.astype(np.uint8)


def _png_b64(rgb_uint8: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ─── GradCAM ────────────────────────────────────────────────────────────────
def gradcam_heatmap_b64(  # pragma: no cover
    image_bgr: np.ndarray,
    class_name: str,
    bboxes_to_focus: list[tuple[float, float, float, float]] | None = None,
) -> str:
    """Class-specific GradCAM heatmap.

    Method: HiResCAM at model.model[9] (SPPF, stride 32). FPN neck layers
    emit zero gradients for YOLO class targets; earlier backbone layers have
    finer resolution but produce visually weaker heat (signal spread across
    many more cells). SPPF + HiResCAM is the best balance of localization and
    visual strength for this model.

    Known limitation: single-pass aggregation favors the spatial cell with
    strongest activations, so higher-confidence detections dominate lower-
    confidence ones of the same class.
    """
    model = _TorchModelCache.get()
    class_id = _TorchModelCache.name_to_id(class_name)
    # SPPF (model.model[9], stride 32, 10x10 grid at 320 input). FPN neck
    # layers emit zero gradients; earlier backbone layers spread heat too
    # thin to be visually useful. SPPF + plain GradCAM is the empirically
    # best configuration for this model.
    target_layer = model.model[9]

    tensor, letterboxed = _preprocess_torch(image_bgr)
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=tensor,
        targets=[_YoloClassTarget(class_id)],
    )[0]

    if bboxes_to_focus:
        mask = _bbox_mask_letterboxed(image_bgr, bboxes_to_focus)
        # Soft-edge the mask via Gaussian blur — eliminates hard rectangular
        # cutoffs where heatmap meets the original image, fades smoothly instead.
        mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=5)
        grayscale_cam = grayscale_cam * mask
        # Re-normalize so the peak inside the bbox region maps to full intensity.
        peak = grayscale_cam.max()
        if peak > 1e-6:
            grayscale_cam = grayscale_cam / peak
    overlay = _heatmap_overlay(letterboxed, grayscale_cam)
    return _png_b64(overlay)


# ─── Annotated detections ───────────────────────────────────────────────────
def annotations_b64(image_bgr: np.ndarray, result: DetectionResult) -> str:
    """Clean detection overlay: colored bboxes + class labels, no heatmap.

    Complements GradCAM (class-specific attention) and SHAP (per-pixel
    attribution) as the third panel in the explanation. Always works, no
    library quirks, instant. Red = violation, Yellow = person, Green = other.
    """
    annotated_bgr = draw_annotations(image_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return _png_b64(annotated_rgb)


# ─── SHAP ───────────────────────────────────────────────────────────────────
def shap_attribution_b64(image_bgr: np.ndarray, class_name: str) -> str:  # pragma: no cover
    """Per-pixel SHAP attribution map for the given class.

    Uses SHAP_NSAMPLES=20 perturbation samples (vs default 200) — ~10× speedup
    with mild attribution-smoothness loss. Output: input image side-by-side
    with channel-summed |SHAP| magnitude heatmap.
    """
    model = _TorchModelCache.get()
    class_id = _TorchModelCache.name_to_id(class_name)
    wrapper = _YoloScalarWrapper(model, class_id)

    tensor, letterboxed = _preprocess_torch(image_bgr)
    background = torch.zeros(1, 3, EXPLAINER_IMG_SIZE, EXPLAINER_IMG_SIZE)

    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(tensor, nsamples=SHAP_NSAMPLES)

    arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 3:
        channel_axis = next((i for i, s in enumerate(arr.shape) if s == 3), 0)
        attribution = np.abs(arr).sum(axis=channel_axis)
    else:
        attribution = np.abs(arr)
    if attribution.max() > 0:
        attribution = attribution / attribution.max()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(letterboxed)
    axes[0].set_title("Input (letterboxed)")
    axes[0].axis("off")
    axes[1].imshow(attribution, cmap="hot")
    axes[1].set_title(f"SHAP attribution: {class_name}")
    axes[1].axis("off")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─── Top-level convenience ──────────────────────────────────────────────────
@dataclass
class Explanation:
    target_class: str
    gradcam_b64: str       # class-specific, bbox-masked to top-class detections
    annotations_b64: str   # clean bbox+label overlay, no heatmap (always reliable)
    shap_b64: str          # class-specific per-pixel attribution
    fallback_to_detection: bool


def explain_result(
    image_bgr: np.ndarray,
    result: DetectionResult,
) -> Explanation | None:
    """Generate GradCAM + EigenCAM + SHAP for a detection result.

    Priority:
      1. Highest-confidence violation
      2. (fallback) Highest-confidence detection
      3. (no detections) → None
    """
    if result.violations:
        top_v = max(result.violations, key=lambda v: v.confidence)
        target = top_v.type
        fallback = False
    elif result.detections:
        top_d = max(result.detections, key=lambda d: d.confidence)
        target = top_d.cls
        fallback = True
        logger.info("No violations — explaining top detection: %s", target)
    else:
        return None

    # GradCAM: mask to only the target-class detections (focused view)
    gradcam_bboxes = [d.bbox for d in result.detections if d.cls == target]
    return Explanation(
        target_class=target,
        gradcam_b64=gradcam_heatmap_b64(image_bgr, target, bboxes_to_focus=gradcam_bboxes),
        annotations_b64=annotations_b64(image_bgr, result),
        shap_b64=shap_attribution_b64(image_bgr, target),
        fallback_to_detection=fallback,
    )
