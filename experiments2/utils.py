"""
experiments2/utils.py
Re-exports all shared utilities from experiments/utils.py so that notebooks
in this folder can simply do `from utils import ...` without path manipulation.

Also adds GradCAM and any new helpers used in the story notebooks.
"""

import sys
from pathlib import Path

# Make sure experiments/ (which contains the real utils.py) is on the path.
_ROOT = Path(__file__).resolve().parent.parent          # project root
_EXP  = _ROOT / "experiments"
for _p in [str(_ROOT), str(_EXP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Re-export everything from experiments/utils.py ───────────────────────────
import importlib as _importlib
_exp_utils = _importlib.import_module("utils")
for _name in dir(_exp_utils):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_exp_utils, _name)

# Keep explicit imports so IDEs can resolve them
from utils import (                                     # noqa: E402,F401
    METRIC_KEYS, NORM_MEAN, NORM_STD,
    NUM_LABELS, LABEL_ORDER,
    TransformSubset,
    set_seed, load_dataset, split_dataset, subsample_subset,
    get_train_transform, get_eval_transform, build_dataloaders,
    plot_per_class_examples, plot_multilabel_examples,
    run_baselines, print_model_info,
    train_model, save_checkpoint, load_checkpoint,
    plot_training_history, plot_multi_arch_histories,
    collect_test_predictions, categorize_predictions,
    show_prediction_examples, plot_per_class_metrics,
    plot_confusion_matrices, plot_prediction_heatmap,
    show_saliency_examples, compute_multilabel_metrics,
    evaluate_predictor, print_metric_table,
    compute_label_prevalence,
)

# ── Additions: GradCAM ───────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM).

    Usage:
        cam = GradCAM(model, target_layer=model.layer3)
        heatmap = cam(image_tensor, class_idx=None)   # image_tensor: (3,H,W) on CPU
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor = None
        self._gradients:   torch.Tensor = None

        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image: torch.Tensor, device,
                 class_idx: int = None) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1]."""
        self.model.eval()
        x = image.unsqueeze(0).to(device).requires_grad_(False)
        # Need gradients for backward pass
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            score = logits.sum()
        else:
            score = logits[0, class_idx]

        self.model.zero_grad()
        score.backward()

        # Weight activations by mean gradient
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu()

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.numpy()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def show_gradcam(indices, images, labels, preds, model, target_layer, title,
                 n=4, device="cpu", norm_mean=None, norm_std=None, label_order=None):
    """Overlay GradCAM heatmap on images for the first n indices."""
    from utils import denorm, labels_to_text
    label_order = label_order or LABEL_ORDER
    norm_mean   = norm_mean or NORM_MEAN
    norm_std    = norm_std  or NORM_STD
    n = min(n, len(indices))
    if n == 0:
        print(f'No examples for "{title}"')
        return

    cam_fn = GradCAM(model, target_layer)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, idx in enumerate(indices[:n].tolist()):
        img  = images[idx]
        hmap = cam_fn(img, device)
        # Upsample heatmap to image size
        import torch.nn.functional as F
        hmap_t = torch.tensor(hmap).unsqueeze(0).unsqueeze(0)
        h, w   = img.shape[1], img.shape[2]
        hmap_t = F.interpolate(hmap_t, size=(h, w), mode="bilinear", align_corners=False)
        hmap_np = hmap_t.squeeze().numpy()

        img_np = denorm(img, norm_mean, norm_std).permute(1, 2, 0).numpy()
        gt_str = labels_to_text(labels[idx], label_order)
        pr_str = labels_to_text(preds[idx],  label_order)

        axes[0][i].imshow(img_np)
        axes[0][i].set_title(f"GT: {gt_str}\nPred: {pr_str}", fontsize=7)
        axes[0][i].axis("off")

        axes[1][i].imshow(img_np)
        axes[1][i].imshow(hmap_np, cmap="jet", alpha=0.45)
        axes[1][i].set_title("GradCAM", fontsize=8)
        axes[1][i].axis("off")

    cam_fn.remove_hooks()
    plt.suptitle(f"GradCAM — {title}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── Stratified split (from ExperimentTransferLearningV2) ─────────────────────
from collections import defaultdict
from torch.utils.data import Subset


def stratified_split_multilabel(dataset, split, seed=42):
    """
    Stratified split for multi-label data by label combination.
    Ensures every rare combo is represented in all three splits.
    """
    combo_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, target = dataset[i]
        combo = tuple(target.int().tolist())
        combo_to_indices[combo].append(i)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for indices in combo_to_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)
        n       = len(indices)
        n_val   = max(1 if n >= 3 else 0, round(split[1] * n))
        n_test  = max(1 if n >= 3 else 0, round(split[2] * n))
        n_train = max(0, n - n_val - n_test)
        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train: n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val:].tolist())

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ── Asymmetric Loss (from ExperimentTransferLearningV2) ──────────────────────
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ben-Baruch et al., 2020).

    Down-weights easy negatives via separate focal parameters for positives
    (gamma_pos) and negatives (gamma_neg), plus a probability shift (clip).
    """

    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
        loss_pos = targets       * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss     = loss_pos + loss_neg
        pt       = probs_pos * targets + probs_neg * (1 - targets)
        gamma    = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss     = loss * (1 - pt).pow(gamma)
        return -loss.mean()
