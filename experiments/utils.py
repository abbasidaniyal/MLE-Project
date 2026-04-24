"""
experiments/utils.py
Shared utilities for all experiment notebooks.
All functions are top-level (no unnecessary nesting).
"""

import sys
import copy
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

sys.path.append("../")
from eval import LABEL_ORDER, CustomDirectoryLayoutDataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = [
    "loss", "exact_match", "hamming_acc", "mean_iou",
    "precision_micro", "recall_micro", "f1_micro",
]
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
NUM_LABELS = len(LABEL_ORDER)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class TransformSubset(Dataset):
    """Wraps a Subset and applies a transform at retrieval time."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, target = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_dataset(base_dir: str) -> CustomDirectoryLayoutDataset:
    """Load the full dataset (no transform applied yet)."""
    dataset = CustomDirectoryLayoutDataset(root=base_dir, transform=None)
    assert len(dataset) > 0, f"Dataset is empty. Check path: {base_dir}"
    return dataset


def split_dataset(full_dataset, split, seed: int = 42):
    """
    Split full_dataset into (train_raw, val_raw, test_raw) subsets.
    split: list of three fractions summing to 1, e.g. [0.8, 0.1, 0.1].
    """
    n_total = len(full_dataset)
    n_train = int(split[0] * n_total)
    n_val   = int(split[1] * n_total)
    n_test  = n_total - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(full_dataset, [n_train, n_val, n_test], generator=generator)


def subsample_subset(subset, fraction: float, seed: int):
    """Return a random fraction of a Subset (useful for quick-test mode)."""
    n_keep = max(1, int(len(subset) * fraction))
    n_drop = len(subset) - n_keep
    keep, _ = random_split(subset, [n_keep, n_drop],
                           generator=torch.Generator().manual_seed(seed))
    return keep


def get_train_transform(image_size: int, norm_mean=None, norm_std=None):
    """Standard augmented transform for training."""
    norm_mean = norm_mean or NORM_MEAN
    norm_std  = norm_std  or NORM_STD
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def get_eval_transform(image_size: int, norm_mean=None, norm_std=None):
    """Deterministic transform for validation / test."""
    norm_mean = norm_mean or NORM_MEAN
    norm_std  = norm_std  or NORM_STD
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def build_dataloaders(train_raw, val_raw, test_raw,
                      train_transform, eval_transform,
                      batch_size: int, num_workers: int = 2):
    """
    Wrap raw subsets with transforms and return (train_loader, val_loader, test_loader).
    """
    train_ds = TransformSubset(train_raw, transform=train_transform)
    val_ds   = TransformSubset(val_raw,   transform=eval_transform)
    test_ds  = TransformSubset(test_raw,  transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def labels_to_text(target, label_order=None):
    """Convert a binary target tensor to a comma-separated label string."""
    label_order = label_order or LABEL_ORDER
    labels = [label_order[i] for i, v in enumerate(target) if int(v) == 1]
    return ", ".join(labels) if labels else "(none)"


def plot_per_class_examples(subset, label_order=None, per_class: int = 3):
    """Plot `per_class` sample images for each class from a raw PIL subset."""
    label_order = label_order or LABEL_ORDER
    selected = {label: [] for label in label_order}
    for image, target in subset:
        target = target.int()
        for i, label in enumerate(label_order):
            if target[i] == 1 and len(selected[label]) < per_class:
                selected[label].append((image.copy(), target.clone()))
        if all(len(v) >= per_class for v in selected.values()):
            break

    fig, axes = plt.subplots(len(label_order), per_class,
                             figsize=(3.2 * per_class, 3.0 * len(label_order)))
    for r, label in enumerate(label_order):
        for c in range(per_class):
            ax = axes[r, c]
            if c < len(selected[label]):
                img, target = selected[label][c]
                ax.imshow(img)
                # Show only the row label as title to avoid long multi-label strings overlapping
                ax.set_title(label, fontsize=9, pad=4)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis("off")
    plt.suptitle("Train examples — samples per class", fontsize=13, y=1.01)
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.3)
    plt.show()


def plot_multilabel_examples(subset, label_order=None, max_items: int = 9):
    """Plot images that have more than one label."""
    label_order = label_order or LABEL_ORDER
    items = []
    for image, target in subset:
        if int(target.sum().item()) > 1:
            items.append((image.copy(), target.clone()))
        if len(items) >= max_items:
            break

    cols = 3
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < len(items):
            img, target = items[idx]
            ax.imshow(img)
            ax.set_title(labels_to_text(target, label_order), fontsize=9)
        ax.axis("off")
    plt.suptitle("Train examples with multiple labels", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_multilabel_metrics(labels, preds, probs=None, logits=None):
    """
    Compute the full set of multi-label classification metrics.
    Returns a dict with keys matching METRIC_KEYS.
    """
    labels = labels.float()
    preds  = preds.float()

    if logits is not None:
        loss = nn.BCEWithLogitsLoss()(logits, labels).item()
    elif probs is not None:
        p    = probs.clamp(1e-6, 1 - 1e-6)
        loss = nn.BCELoss()(p, labels).item()
    else:
        loss = float("nan")

    exact_match = (preds == labels).all(dim=1).float().mean().item()
    hamming_acc = (preds == labels).float().mean().item()

    intersection = (preds * labels).sum(dim=1)
    union = ((preds + labels) > 0).float().sum(dim=1)
    iou  = torch.where(union > 0, intersection / union, torch.ones_like(union))
    mean_iou = iou.mean().item()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    precision_micro = (tp / (tp + fp + 1e-8)).item()
    recall_micro    = (tp / (tp + fn + 1e-8)).item()
    f1_micro        = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    return {
        "loss": loss, "exact_match": exact_match, "hamming_acc": hamming_acc,
        "mean_iou": mean_iou, "precision_micro": precision_micro,
        "recall_micro": recall_micro, "f1_micro": f1_micro,
    }


def evaluate_predictor(data_loader, predict_fn, device, threshold: float = 0.5):
    """
    Evaluate a predict_fn(images, threshold) -> (preds, probs, logits) over a DataLoader.
    Returns a metrics dict.
    """
    all_labels, all_preds, all_probs, all_logits = [], [], [], []
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds, probs, logits = predict_fn(images, threshold=threshold)
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_logits.append(logits.cpu())
    return compute_multilabel_metrics(
        torch.cat(all_labels), torch.cat(all_preds),
        probs=torch.cat(all_probs), logits=torch.cat(all_logits),
    )


def print_metric_table(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    for k in METRIC_KEYS:
        print(f"  {k:<16} {metrics[k]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def compute_label_prevalence(train_loader):
    """Compute per-label fraction of positive samples in the training set."""
    all_targets = torch.cat([targets for _, targets in train_loader], dim=0)
    return all_targets.mean(dim=0)  # (C,)


def make_topk_predictor(label_prevalence, k: int, num_labels: int, device):
    """Return a predict_fn that always predicts the k most frequent labels."""
    topk_idx   = label_prevalence.argsort(descending=True)[:k]
    fixed_pred = torch.zeros(num_labels, device=device)
    fixed_pred[topk_idx] = 1.0

    def predict_fn(images, threshold=0.5):
        bsz    = images.shape[0]
        preds  = fixed_pred.unsqueeze(0).expand(bsz, -1).clone()
        probs  = preds * 0.9 + (1 - preds) * 0.1
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        return preds, probs, logits

    return predict_fn


def make_random_predictor(num_labels: int):
    """Return a predict_fn that samples each label independently at p=0.5."""

    def predict_fn(images, threshold=0.5):
        bsz    = images.shape[0]
        probs  = torch.full((bsz, num_labels), 0.5, device=images.device)
        probs  = probs + 0.01 * torch.randn_like(probs)
        preds  = (probs >= threshold).float()
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        return preds, probs, logits

    return predict_fn


def run_baselines(train_loader, val_loader, test_loader, num_labels: int, device):
    """
    Evaluate top-1/2/3 frequency and random baselines.
    Prints val metrics, picks the best by F1, reports it on test.
    Returns (best_name, val_metrics_dict, test_metrics_dict).
    """
    prevalence = compute_label_prevalence(train_loader)

    print("Train label prevalence (sorted):")
    for rank, i in enumerate(prevalence.argsort(descending=True).tolist()):
        print(f"  #{rank+1:<2} {LABEL_ORDER[i]:<12}  {prevalence[i].item():.3f}")

    baselines = {
        "top-1 freq": make_topk_predictor(prevalence, 1, num_labels, device),
        "top-2 freq": make_topk_predictor(prevalence, 2, num_labels, device),
        "top-3 freq": make_topk_predictor(prevalence, 3, num_labels, device),
        "random":     make_random_predictor(num_labels),
    }

    best_name = None
    best_f1   = -1.0
    val_metrics_all = {}
    for name, fn in baselines.items():
        m = evaluate_predictor(val_loader, fn, device)
        val_metrics_all[name] = m
        print_metric_table(f"{name}  (Val)", m)
        if m["f1_micro"] > best_f1:
            best_f1  = m["f1_micro"]
            best_name = name

    print(f'\n>>> Best baseline: "{best_name}"  (val F1={best_f1:.4f})')
    test_metrics = evaluate_predictor(test_loader, baselines[best_name], device)
    print_metric_table(f'Best baseline "{best_name}" (Test)', test_metrics)
    return best_name, val_metrics_all[best_name], test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

def print_model_info(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / 1024 / 1024
    size_str  = f"{size_mb:.2f} MB" if size_mb < 1024 else f"{size_mb / 1024:.3f} GB"
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable params : {trainable:>12,}  ({100 * trainable / total:.1f}%)")
    print(f"  Model size       : {size_str}  (float32 weights)")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(create_model_fn, num_labels: int, train_loader, val_loader, device,
                lr: float = 1e-3, weight_decay: float = 1e-3, max_epochs: int = 30,
                warmup_epochs: int = 5, early_stop_patience: int = 5,
                lr_reduce_patience: int = 3, lr_reduce_factor: float = 0.5,
                grad_clip: float = 1.0, threshold: float = 0.5,
                criterion=None):
    """
    Train a fresh model (built by create_model_fn(num_labels)) with:
      - Linear LR warmup for `warmup_epochs` epochs
      - ReduceLROnPlateau (maximising val F1) after warmup
      - Early stopping with fallback to best weights
    Returns: (best_state_dict, best_val_f1, history, epochs_run)
    """
    criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
    model = create_model_fn(num_labels).to(device)
    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=lr_reduce_factor, patience=lr_reduce_patience
    )

    history     = {k: {"train": [], "val": []} for k in METRIC_KEYS}
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    epoch       = 0

    for epoch in range(1, max_epochs + 1):
        if epoch <= warmup_epochs:
            for pg in opt.param_groups:
                pg["lr"] = lr * epoch / warmup_epochs

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_l, tr_p, tr_pr, tr_lg = [], [], [], []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            opt.zero_grad()
            logits = model(images)
            loss   = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
            tr_l.append(targets.cpu());  tr_p.append(preds.cpu())
            tr_pr.append(probs.cpu());   tr_lg.append(logits.detach().cpu())

        train_metrics = compute_multilabel_metrics(
            torch.cat(tr_l), torch.cat(tr_p),
            probs=torch.cat(tr_pr), logits=torch.cat(tr_lg),
        )

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_l, val_p, val_pr, val_lg = [], [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                probs  = torch.sigmoid(logits)
                preds  = (probs >= threshold).float()
                val_l.append(labels.cpu());  val_p.append(preds.cpu())
                val_pr.append(probs.cpu());  val_lg.append(logits.cpu())

        val_metrics = compute_multilabel_metrics(
            torch.cat(val_l), torch.cat(val_p),
            probs=torch.cat(val_pr), logits=torch.cat(val_lg),
        )

        for k in METRIC_KEYS:
            history[k]["train"].append(train_metrics[k])
            history[k]["val"].append(val_metrics[k])

        val_f1 = val_metrics["f1_micro"]
        if epoch > warmup_epochs:
            scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1

        lr_now = opt.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:>2}/{max_epochs}  [lr={lr_now:.2e}]")
        print(f"  {'Metric':<20} {'Train':>8}  {'Val':>8}")
        print(f"  {'-'*40}")
        for k in METRIC_KEYS:
            print(f"  {k:<20} {train_metrics[k]:>8.4f}  {val_metrics[k]:>8.4f}")

        if no_improve >= early_stop_patience:
            print(f"\n[Early stop] No val F1 improvement for {early_stop_patience} epochs.")
            break

    return best_state, best_val_f1, history, epoch


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state_dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    torch.save(state_dict, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(create_model_fn, num_labels: int, path, device):
    """Build model, load weights, set eval mode and return."""
    model = create_model_fn(num_labels).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, epochs_run: int,
                          experiment_name: str = "", lr: float = 0, weight_decay: float = 0):
    """Plot train/val curves for every metric in METRIC_KEYS."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    er   = range(1, epochs_run + 1)
    for i, k in enumerate(METRIC_KEYS):
        axes[i].plot(er, history[k]["train"], label="train")
        axes[i].plot(er, history[k]["val"],   label="val")
        axes[i].set_title(k)
        axes[i].legend()
        axes[i].set_xlabel("Epoch")
    axes[-1].axis("off")
    plt.suptitle(f"{experiment_name}  |  lr={lr:.0e}  wd={weight_decay:.0e}", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_multi_arch_histories(all_histories, experiment_name: str = ""):
    """Plot training curves for multiple architectures on the same axes."""
    colors = ["steelblue", "darkorange", "green", "purple", "red"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for (arch, history), color in zip(all_histories.items(), colors):
        er = range(1, len(history["loss"]["train"]) + 1)
        for i, k in enumerate(METRIC_KEYS):
            axes[i].plot(er, history[k]["train"], label=f"{arch} train",
                         color=color, linestyle="--", alpha=0.7)
            axes[i].plot(er, history[k]["val"],   label=f"{arch} val", color=color)
            axes[i].set_title(k)
            axes[i].set_xlabel("Epoch")
    for ax in axes[:len(METRIC_KEYS)]:
        ax.legend(fontsize=7)
    axes[-1].axis("off")
    plt.suptitle(f"{experiment_name} — training curves per architecture", fontsize=12)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction analysis
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_predictions(model: nn.Module, test_loader, device, threshold: float = 0.5):
    """
    Run inference over test_loader.
    Returns (images, labels, preds, probs) as CPU tensors.
    """
    all_images, all_labels, all_preds, all_probs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probs  = torch.sigmoid(logits)
            preds  = (probs >= threshold).float()
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
    return (
        torch.cat(all_images),
        torch.cat(all_labels),
        torch.cat(all_preds),
        torch.cat(all_probs),
    )


def categorize_predictions(labels, preds):
    """
    Split test indices into three buckets.
    Returns (correct_idx, partial_idx, incorrect_idx) as 1-D tensors.
    """
    correct_mask  = (preds == labels).all(dim=1)
    incorrect_mask = ((preds * labels).sum(dim=1) == 0) & ~correct_mask
    partial_mask   = ~correct_mask & ~incorrect_mask
    correct_idx   = correct_mask.nonzero(as_tuple=True)[0]
    partial_idx   = partial_mask.nonzero(as_tuple=True)[0]
    incorrect_idx = incorrect_mask.nonzero(as_tuple=True)[0]
    print(f"Fully correct    : {len(correct_idx):>5} / {len(labels)}")
    print(f"Partially correct: {len(partial_idx):>5} / {len(labels)}")
    print(f"Fully incorrect  : {len(incorrect_idx):>5} / {len(labels)}")
    return correct_idx, partial_idx, incorrect_idx


def denorm(t: torch.Tensor, norm_mean=None, norm_std=None) -> torch.Tensor:
    """Reverse ImageNet normalisation for a (3, H, W) tensor."""
    norm_mean = norm_mean or NORM_MEAN
    norm_std  = norm_std  or NORM_STD
    _mean = torch.tensor(norm_mean).view(3, 1, 1)
    _std  = torch.tensor(norm_std).view(3, 1, 1)
    return (t * _std + _mean).clamp(0, 1)


def show_prediction_examples(indices, images, labels, preds, title: str, n: int = 4,
                              norm_mean=None, norm_std=None, label_order=None):
    """Display n example predictions (ground truth vs predicted labels)."""
    label_order = label_order or LABEL_ORDER
    n = min(n, len(indices))
    if n == 0:
        print(f'No examples for "{title}"')
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, idx in enumerate(indices[:n].tolist()):
        img_np   = denorm(images[idx], norm_mean, norm_std).permute(1, 2, 0).numpy()
        true_lbl = labels_to_text(labels[idx], label_order)
        pred_lbl = labels_to_text(preds[idx],  label_order)
        axes[i].imshow(img_np)
        axes[i].set_title(f"GT:   {true_lbl}\nPred: {pred_lbl}", fontsize=8)
        axes[i].axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(labels, preds, label_order=None):
    """Print per-class accuracy/precision/recall/F1 table and F1 bar chart."""
    label_order = label_order or LABEL_ORDER
    num_cls = len(label_order)
    L = labels.float()
    P = preds.float()

    tp = ((P == 1) & (L == 1)).sum(dim=0).float()
    fp = ((P == 1) & (L == 0)).sum(dim=0).float()
    fn = ((P == 0) & (L == 1)).sum(dim=0).float()
    tn = ((P == 0) & (L == 0)).sum(dim=0).float()

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc  = (tp + tn) / len(L)

    print(f"{'Label':<14} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("-" * 70)
    for i, lbl in enumerate(label_order):
        print(f"{lbl:<14} {acc[i].item():>6.3f} {prec[i].item():>6.3f} "
              f"{rec[i].item():>6.3f} {f1[i].item():>6.3f}  "
              f"{int(tp[i]):>5} {int(fp[i]):>5} {int(fn[i]):>5} {int(tn[i]):>5}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(num_cls), f1.numpy(), color="steelblue")
    ax.set_xticks(range(num_cls))
    ax.set_xticklabels(label_order, rotation=40, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on Test Set")
    ax.axhline(f1.mean().item(), color="red", linestyle="--",
               label=f"mean F1 = {f1.mean().item():.3f}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(labels, preds, label_order=None):
    """Show a 2×2 binary confusion matrix for each class."""
    label_order = label_order or LABEL_ORDER
    num_cls = len(label_order)
    L = labels.float()
    P = preds.float()

    tp = ((P == 1) & (L == 1)).sum(dim=0).float()
    fp = ((P == 1) & (L == 0)).sum(dim=0).float()
    fn = ((P == 0) & (L == 1)).sum(dim=0).float()
    tn = ((P == 0) & (L == 0)).sum(dim=0).float()

    n_cols = 4
    n_rows = (num_cls + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    for i, lbl in enumerate(label_order):
        cm = np.array([[int(tn[i]), int(fp[i])],
                       [int(fn[i]), int(tp[i])]])
        axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(lbl, fontsize=10, fontweight="bold")
        axes[i].set_xticks([0, 1]); axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        axes[i].set_yticklabels(["GT 0",  "GT 1"],  fontsize=8)
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, str(cm[r, c]), ha="center", va="center",
                             fontsize=11,
                             color="white" if cm[r, c] > cm.max() * 0.5 else "black")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Per-class Binary Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_prediction_heatmap(labels, preds, label_order=None):
    """
    Co-occurrence heatmap: co_matrix[i, j] = samples where label i is in GT
    and label j is predicted.
    """
    label_order = label_order or LABEL_ORDER
    num_cls     = len(label_order)
    co_matrix   = (labels.float().T @ preds.float()).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(co_matrix, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="count")
    ax.set_xticks(range(num_cls)); ax.set_yticks(range(num_cls))
    ax.set_xticklabels(label_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_order, fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("GT label")
    ax.set_title("GT-vs-Prediction co-occurrence (Test Set)\n"
                 "Diagonal = true positives; off-diagonal = FP / FN patterns", fontsize=11)
    for i in range(num_cls):
        for j in range(num_cls):
            v = int(co_matrix[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if co_matrix[i, j] > co_matrix.max() * 0.6 else "black")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Saliency maps
# ─────────────────────────────────────────────────────────────────────────────

def compute_saliency(model: nn.Module, image_tensor: torch.Tensor, device) -> torch.Tensor:
    """
    Gradient-based saliency for a single (3, H, W) image tensor.
    Returns a (H, W) tensor with values in [0, 1].
    """
    model.eval()
    inp    = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    logits = model(inp)
    score  = logits.sum()
    model.zero_grad()
    score.backward()
    saliency    = inp.grad.data.abs().squeeze(0)
    saliency, _ = saliency.max(dim=0)
    saliency    = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.cpu()


def show_saliency_examples(indices, images, labels, preds, model: nn.Module,
                           title: str, n: int = 3,
                           norm_mean=None, norm_std=None, device=None, label_order=None):
    """Show original + saliency overlay side-by-side for n examples."""
    label_order = label_order or LABEL_ORDER
    device      = device or torch.device("cpu")
    n = min(n, len(indices))
    if n == 0:
        print(f'No examples for "{title}"')
        return
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]
    for i, idx in enumerate(indices[:n].tolist()):
        img_tensor = images[idx]
        saliency   = compute_saliency(model, img_tensor, device)
        img_np     = denorm(img_tensor, norm_mean, norm_std).permute(1, 2, 0).numpy()
        true_lbl   = labels_to_text(labels[idx], label_order)
        pred_lbl   = labels_to_text(preds[idx],  label_order)

        axes[i][0].imshow(img_np)
        axes[i][0].set_title(f"GT:   {true_lbl}\nPred: {pred_lbl}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(img_np)
        axes[i][1].imshow(saliency.numpy(), cmap="hot", alpha=0.55)
        axes[i][1].set_title("Saliency overlay", fontsize=8)
        axes[i][1].axis("off")

    plt.suptitle(f"Saliency Maps — {title}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
