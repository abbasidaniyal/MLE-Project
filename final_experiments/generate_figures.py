"""
generate_figures.py
-------------------
Generates publication-quality PNG figures for the data exploration section.
Run from the project root:   python final_experiments/generate_figures.py
Figures are saved to:        final_experiments/figures/
"""

import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, ".")

from eval import LABEL_ORDER, CustomDirectoryLayoutDataset
from utils import (
    set_seed, load_dataset, split_dataset,
    get_train_transform, get_eval_transform, build_dataloaders,
    compute_label_prevalence, NUM_LABELS,
)

# ── Config ──────────────────────────────────────────────────────────────────
SEED       = 42
BASE_DIR   = "data/aggregated"
IMAGE_SIZE = 128
BATCH_SIZE = 128
SPLIT      = [0.7, 0.15, 0.15]
OUT_DIR    = Path("final_experiments/figures")
DPI        = 200      # publication quality
PAPER_RC   = {        # clean, minimal style
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "figure.dpi":        DPI,
}

set_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset …")
full_dataset = load_dataset(BASE_DIR)
print(f"  {len(full_dataset)} images  ·  {NUM_LABELS} classes")

train_raw, val_raw, test_raw = split_dataset(full_dataset, SPLIT, SEED)
train_transform = get_train_transform(IMAGE_SIZE)
eval_transform  = get_eval_transform(IMAGE_SIZE)
train_loader, val_loader, test_loader = build_dataloaders(
    train_raw, val_raw, test_raw, train_transform, eval_transform,
    batch_size=BATCH_SIZE,
)

# ── Helper ───────────────────────────────────────────────────────────────────
NICE_LABEL = {l: l.capitalize() for l in LABEL_ORDER}

COLOR_SEQ = plt.cm.tab20.colors          # 20 distinct colours


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Per-class image gallery   (horizontal: 12 cols × per_class rows)
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_class_gallery(subset, per_class: int = 2, save_path: Path = None):
    selected = {lbl: [] for lbl in LABEL_ORDER}
    used = set()  # track indices already assigned to avoid reusing the same image
    for idx in range(len(subset)):
        if all(len(v) >= per_class for v in selected.values()):
            break
        if idx in used:
            continue
        image, target = subset[idx]
        target_int = target.int()
        for i, lbl in enumerate(LABEL_ORDER):
            if target_int[i] == 1 and len(selected[lbl]) < per_class:
                selected[lbl].append(image.copy())
                used.add(idx)
                break  # assign each image to at most one class

    # horizontal layout: one column per class, per_class image rows + 1 label row
    nrows   = per_class + 1   # +1 for the class-name header row
    ncols   = NUM_LABELS
    cell    = 1.6             # inches per cell
    label_h = 0.35            # fractional height of the header row

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(ncols * cell, per_class * cell + 0.55))

        outer = gridspec.GridSpec(
            nrows, ncols,
            figure=fig,
            wspace=0.05, hspace=0.06,
            left=0.01, right=0.99,
            top=0.93,  bottom=0.01,
            height_ratios=[label_h] + [1] * per_class,
        )

        for c, lbl in enumerate(LABEL_ORDER):
            # ── column header (class name) ─────────────────────────────────
            ax_hdr = fig.add_subplot(outer[0, c])
            ax_hdr.axis("off")
            ax_hdr.text(
                0.5, 0.0, NICE_LABEL[lbl],
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                transform=ax_hdr.transAxes,
            )

            # ── image rows ────────────────────────────────────────────────
            for r in range(per_class):
                ax = fig.add_subplot(outer[r + 1, c])
                if r < len(selected[lbl]):
                    ax.imshow(selected[lbl][r])
                else:
                    ax.set_facecolor("#f0f0f0")
                ax.axis("off")

        fig.suptitle("Sample images per class", fontsize=11, y=0.98, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Multi-label gallery   (3 rows × 6 images)
# ─────────────────────────────────────────────────────────────────────────────
def fig_multilabel_gallery(subset, n_items: int = 18, ncols: int = 6,
                           save_path: Path = None):
    items = []
    for image, target in subset:
        if int(target.sum().item()) >= 2:
            items.append((image.copy(), target.clone()))
        if len(items) >= n_items:
            break

    nrows = (len(items) + ncols - 1) // ncols
    cell  = 1.8
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell, nrows * cell + 0.4),
        )
        axes = np.array(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            if idx < len(items):
                img, tgt = items[idx]
                lbls = [NICE_LABEL[LABEL_ORDER[i]] for i, v in enumerate(tgt) if int(v) == 1]
                ax.imshow(img)
                # wrap long label strings
                title = " · ".join(lbls)
                ax.set_title(title, fontsize=6.5, pad=2.5, color="#222222")
            ax.axis("off")

        fig.suptitle("Examples with multiple labels", fontsize=11,
                     fontweight="bold", y=1.01)
        plt.tight_layout(pad=0.3, h_pad=0.7, w_pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Label prevalence  (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_label_prevalence(train_loader, save_path: Path = None):
    prevalence = compute_label_prevalence(train_loader)
    order      = prevalence.argsort(descending=True)
    labels_s   = [NICE_LABEL[LABEL_ORDER[i]] for i in order]
    values_s   = prevalence[order].numpy()
    colors     = [COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(labels_s))]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        bars = ax.barh(labels_s[::-1], values_s[::-1], color=colors[::-1],
                       edgecolor="white", linewidth=0.4, height=0.65)
        for bar, val in zip(bars, values_s[::-1]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left", fontsize=8)
        ax.set_xlim(0, 1.06)
        ax.set_xlabel("Fraction of images")
        ax.set_title("Label prevalence (training split)", fontweight="bold")
        ax.axvline(0.5, color="#aaaaaa", lw=0.8, linestyle="--")
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Labels-per-image distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_labels_per_image(full_dataset, save_path: Path = None):
    label_counts = Counter(int(t.sum()) for _, t in full_dataset)
    xs     = sorted(label_counts)
    ys     = [label_counts[x] for x in xs]
    total  = sum(ys)
    colors = [COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(xs))]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bars = ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.4, width=0.6)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + total * 0.004,
                    f"{y:,}\n({100*y/total:.0f}%)",
                    ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_xlabel("Number of labels per image")
        ax.set_ylabel("Image count")
        ax.set_title("Distribution of labels per image", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_ylim(0, max(ys) * 1.18)
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Label co-occurrence matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig_cooccurrence(full_dataset, save_path: Path = None):
    all_targets = torch.stack([tgt for _, tgt in full_dataset])
    cooc = (all_targets.T @ all_targets) / len(all_targets)
    nice = [NICE_LABEL[l] for l in LABEL_ORDER]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 5.8))
        im = ax.imshow(cooc.numpy(), cmap="Blues", vmin=0, vmax=cooc.max().item(),
                       aspect="equal")
        ax.set_xticks(range(NUM_LABELS))
        ax.set_yticks(range(NUM_LABELS))
        ax.set_xticklabels(nice, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(nice, fontsize=8)
        ax.set_title("Label co-occurrence (normalised)", fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

        # annotate cells whose value ≥ 0.10
        thresh = cooc.max().item() * 0.5
        for i in range(NUM_LABELS):
            for j in range(NUM_LABELS):
                v = cooc[i, j].item()
                if v >= 0.05:
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center",
                            fontsize=6,
                            color="white" if v >= thresh else "#333333")
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Dataset split sizes
# ─────────────────────────────────────────────────────────────────────────────
def fig_dataset_splits(train_raw, val_raw, test_raw, save_path: Path = None):
    names  = ["Train", "Validation", "Test"]
    sizes  = [len(train_raw), len(val_raw), len(test_raw)]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    total  = sum(sizes)

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(names, sizes, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
        for bar, n in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    n + total * 0.005,
                    f"{n:,}\n({100*n/total:.0f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Images")
        ax.set_title("Dataset split (70 / 15 / 15)", fontweight="bold")
        ax.set_ylim(0, max(sizes) * 1.2)
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating figures …")

    fig_per_class_gallery(
        full_dataset,
        per_class=2,
        save_path=OUT_DIR / "fig1_per_class_gallery.png",
    )

    fig_multilabel_gallery(
        full_dataset,
        n_items=18, ncols=6,
        save_path=OUT_DIR / "fig2_multilabel_gallery.png",
    )

    fig_label_prevalence(
        train_loader,
        save_path=OUT_DIR / "fig3_label_prevalence.png",
    )

    fig_labels_per_image(
        full_dataset,
        save_path=OUT_DIR / "fig4_labels_per_image.png",
    )

    fig_cooccurrence(
        full_dataset,
        save_path=OUT_DIR / "fig5_label_cooccurrence.png",
    )

    fig_dataset_splits(
        train_raw, val_raw, test_raw,
        save_path=OUT_DIR / "fig6_dataset_splits.png",
    )

    print(f"\nAll figures saved to  {OUT_DIR.resolve()}/")
