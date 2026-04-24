#!/usr/bin/env python3
"""Generate all final_experiments/*.ipynb files programmatically."""

import json
from pathlib import Path

OUT = Path(__file__).parent


# ─── helpers ──────────────────────────────────────────────────────────────────

def md(src, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}

def code(src, cell_id):
    return {
        "cell_type": "code", "id": cell_id,
        "metadata": {}, "execution_count": None, "outputs": [], "source": src,
    }

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

def save(filename, notebook):
    path = OUT / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"  created: {path}")


# ─── shared snippets ──────────────────────────────────────────────────────────

COMMON_IMPORTS = """\
import sys
import time
sys.path.insert(0, "..")
sys.path.insert(0, "../experiments")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

from eval import LABEL_ORDER
from utils import (
    set_seed, load_dataset, split_dataset,
    get_train_transform, get_eval_transform, build_dataloaders,
    train_model, save_checkpoint, load_checkpoint,
    plot_training_history, print_model_info,
    compute_multilabel_metrics, evaluate_predictor,
    print_metric_table, NUM_LABELS, METRIC_KEYS,
)

SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Labels ({NUM_LABELS}): {LABEL_ORDER}")
"""

def data_loading_block(image_size, batch_size, experiment_name):
    return f"""\
BASE_DIR        = "../data/aggregated"
IMAGE_SIZE      = {image_size}
BATCH_SIZE      = {batch_size}
SPLIT           = [0.7, 0.15, 0.15]
CHECKPOINT_DIR  = Path("../checkpoints")
EXPERIMENT_NAME = "{experiment_name}"
MODEL_PATH      = CHECKPOINT_DIR / f"final_{{EXPERIMENT_NAME}}.pth"

full_dataset = load_dataset(BASE_DIR)
train_raw, val_raw, test_raw = split_dataset(full_dataset, SPLIT, SEED)

train_transform = get_train_transform(IMAGE_SIZE)
eval_transform  = get_eval_transform(IMAGE_SIZE)
train_loader, val_loader, test_loader = build_dataloaders(
    train_raw, val_raw, test_raw, train_transform, eval_transform,
    batch_size=BATCH_SIZE,
)
print(f"Train: {{len(train_raw)}}  |  Val: {{len(val_raw)}}  |  Test: {{len(test_raw)}}")
"""

GRID_SEARCH = """\
GRID = [
    {"lr": 1e-3, "wd": 1e-4},
    {"lr": 1e-3, "wd": 1e-3},
    {"lr": 3e-4, "wd": 1e-4},
    {"lr": 1e-4, "wd": 1e-4},
]

grid_results = []
for cfg in GRID:
    print(f"\\n--- lr={cfg['lr']:.0e}  wd={cfg['wd']:.0e} ---")
    state, val_f1, _, epochs_run = train_model(
        create_model, NUM_LABELS, train_loader, val_loader, DEVICE,
        lr=cfg["lr"], weight_decay=cfg["wd"],
        max_epochs=20, warmup_epochs=2, early_stop_patience=5,
    )
    grid_results.append({**cfg, "val_f1": val_f1, "state": state, "epochs": epochs_run})
    print(f"  => val F1: {val_f1:.4f}")

grid_results.sort(key=lambda x: x["val_f1"], reverse=True)
best = grid_results[0]
print(f"\\nBest config: lr={best['lr']:.0e}  wd={best['wd']:.0e}  val_F1={best['val_f1']:.4f}")

rows = [{"lr": c["lr"], "wd": c["wd"], "val_f1": round(c["val_f1"], 4), "epochs": c["epochs"]}
        for c in grid_results]
print(pd.DataFrame(rows).to_string(index=False))
"""

FINAL_TRAIN = """\
t0 = time.time()
best_state, best_val_f1, history, epochs_run = train_model(
    create_model, NUM_LABELS, train_loader, val_loader, DEVICE,
    lr=best["lr"], weight_decay=best["wd"],
    max_epochs=60, warmup_epochs=5, early_stop_patience=10,
)
training_time = time.time() - t0
print(f"\\nBest val F1: {best_val_f1:.4f}  |  Epochs: {epochs_run}  |  Time: {training_time:.1f}s")

save_checkpoint(best_state, MODEL_PATH)
plot_training_history(history, epochs_run, EXPERIMENT_NAME, best["lr"], best["wd"])
"""

EVAL_BLOCK = """\
model = load_checkpoint(create_model, NUM_LABELS, MODEL_PATH, DEVICE)
model.eval()

def _predict(images, threshold=0.5):
    with torch.no_grad():
        logits = model(images)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).float()
    return preds, probs, logits

val_metrics  = evaluate_predictor(val_loader,  _predict, DEVICE)
test_metrics = evaluate_predictor(test_loader, _predict, DEVICE)

rows = [
    {"split": "val",  **{k: round(val_metrics[k],  4) for k in METRIC_KEYS}},
    {"split": "test", **{k: round(test_metrics[k], 4) for k in METRIC_KEYS}},
]
df = pd.DataFrame(rows).set_index("split")
print(df.to_string())
"""

MODEL_INFO_SUFFIX = """\

print("\\nModel summary:")
print_model_info(create_model(NUM_LABELS))
print(f"Training time : {training_time:.1f}s")
"""


# ─── notebook 01: data + baselines ───────────────────────────────────────────

def make_nb01():
    cells = [
        md("# 01 — Data Exploration and Baselines\n\nMulti-label image classification · 12 classes · 128×128 RGB", "nb01-title"),

        code("""\
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../experiments")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import Counter

from eval import LABEL_ORDER
from utils import (
    set_seed, load_dataset, split_dataset,
    get_train_transform, get_eval_transform, build_dataloaders,
    plot_per_class_examples, plot_multilabel_examples,
    compute_label_prevalence,
    make_topk_predictor, make_random_predictor,
    evaluate_predictor, print_metric_table,
    NUM_LABELS, METRIC_KEYS,
)

SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Labels ({NUM_LABELS}): {LABEL_ORDER}")
""", "nb01-imports"),

        code("""\
BASE_DIR   = "../data/aggregated"
IMAGE_SIZE = 128
BATCH_SIZE = 128
SPLIT      = [0.7, 0.15, 0.15]
""", "nb01-config"),

        md("## 1. Data Visualization", "nb01-sec1"),

        code("""\
full_dataset = load_dataset(BASE_DIR)
print(f"Total images: {len(full_dataset)}")
""", "nb01-load"),

        code("plot_per_class_examples(full_dataset, LABEL_ORDER, per_class=3)", "nb01-per-class"),

        code("plot_multilabel_examples(full_dataset, LABEL_ORDER, max_items=9)", "nb01-multilabel"),

        md("## 2. Dataset Statistics", "nb01-sec2"),

        code("""\
all_targets = torch.stack([target for _, target in full_dataset])
label_freq  = all_targets.mean(dim=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(LABEL_ORDER, label_freq.numpy())
ax.set_xlabel("Label")
ax.set_ylabel("Frequency")
ax.set_title("Label prevalence")
ax.set_ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

df_freq = pd.DataFrame({"label": LABEL_ORDER, "frequency": label_freq.numpy().round(3)})
print(df_freq.sort_values("frequency", ascending=False).to_string(index=False))
""", "nb01-label-freq"),

        code("""\
label_counts = Counter(int(t.sum()) for _, t in full_dataset)

fig, ax = plt.subplots(figsize=(8, 4))
xs = sorted(label_counts)
ax.bar(xs, [label_counts[x] for x in xs])
ax.set_xlabel("Labels per image")
ax.set_ylabel("Images")
ax.set_title("Distribution of labels per image")
for x in xs:
    ax.text(x, label_counts[x] + 5, str(label_counts[x]), ha="center", fontsize=9)
plt.tight_layout()
plt.show()
""", "nb01-label-dist"),

        code("""\
cooc = (all_targets.T @ all_targets) / len(all_targets)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cooc.numpy(), cmap="Blues", vmin=0, vmax=cooc.max().item())
ax.set_xticks(range(NUM_LABELS))
ax.set_yticks(range(NUM_LABELS))
ax.set_xticklabels(LABEL_ORDER, rotation=45, ha="right")
ax.set_yticklabels(LABEL_ORDER)
ax.set_title("Label co-occurrence (normalized)")
plt.colorbar(im)
plt.tight_layout()
plt.show()
""", "nb01-cooc"),

        md("## 3. Data Splitting (70 / 15 / 15)", "nb01-sec3"),

        code("""\
train_raw, val_raw, test_raw = split_dataset(full_dataset, SPLIT, SEED)

train_transform = get_train_transform(IMAGE_SIZE)
eval_transform  = get_eval_transform(IMAGE_SIZE)
train_loader, val_loader, test_loader = build_dataloaders(
    train_raw, val_raw, test_raw, train_transform, eval_transform,
    batch_size=BATCH_SIZE,
)
print(f"Train: {len(train_raw)}  |  Val: {len(val_raw)}  |  Test: {len(test_raw)}")
""", "nb01-split"),

        md("## 4. Baselines", "nb01-sec4"),

        code("""\
def make_prior_weighted_predictor(label_prevalence, num_labels):
    \"\"\"Each label sampled independently with p = training prevalence.\"\"\"
    def predict_fn(images, threshold=0.5):
        bsz   = images.shape[0]
        probs = label_prevalence.unsqueeze(0).expand(bsz, -1).to(images.device).clone()
        preds = torch.bernoulli(probs)
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        return preds, probs, logits
    return predict_fn

def make_all_positive_predictor(num_labels):
    \"\"\"Always predicts every class as positive.\"\"\"
    def predict_fn(images, threshold=0.5):
        bsz   = images.shape[0]
        preds  = torch.ones(bsz, num_labels, device=images.device)
        probs  = torch.full((bsz, num_labels), 0.95, device=images.device)
        logits = torch.logit(probs)
        return preds, probs, logits
    return predict_fn
""", "nb01-baseline-defs"),

        code("""\
prevalence = compute_label_prevalence(train_loader)

baselines = {
    "top-1 freq":      make_topk_predictor(prevalence, 1, NUM_LABELS, DEVICE),
    "top-2 freq":      make_topk_predictor(prevalence, 2, NUM_LABELS, DEVICE),
    "top-3 freq":      make_topk_predictor(prevalence, 3, NUM_LABELS, DEVICE),
    "random uniform":  make_random_predictor(NUM_LABELS),
    "random weighted": make_prior_weighted_predictor(prevalence, NUM_LABELS),
    "all positive":    make_all_positive_predictor(NUM_LABELS),
}

results = {}
for name, fn in baselines.items():
    results[name] = evaluate_predictor(val_loader, fn, DEVICE)
    print_metric_table(f"{name} (val)", results[name])
""", "nb01-run-baselines"),

        md("## 5. Results Summary", "nb01-sec5"),

        code("""\
import pandas as pd

rows = []
for name, m in results.items():
    rows.append({"baseline": name, **{k: round(m[k], 4) for k in METRIC_KEYS}})

df = pd.DataFrame(rows).set_index("baseline")
print(df.to_string())
""", "nb01-results"),
    ]
    return nb(cells)


# ─── notebook 02: Small CNN ───────────────────────────────────────────────────

def make_nb02():
    model_def = """\
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding="same"), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding="same"), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding="same"), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding="same"), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

def create_model(num_labels):
    return SmallCNN(num_classes=num_labels)

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 02 — Small CNN (from scratch)", "nb02-title"),
        code(COMMON_IMPORTS, "nb02-imports"),
        code(data_loading_block(128, 128, "small_cnn"), "nb02-data"),
        md("## Model Definition", "nb02-model-sec"),
        code(model_def, "nb02-model"),
        md("## Grid Search (LR × WD)", "nb02-gs-sec"),
        code(GRID_SEARCH, "nb02-grid"),
        md("## Final Training", "nb02-train-sec"),
        code(FINAL_TRAIN, "nb02-train"),
        md("## Evaluation", "nb02-eval-sec"),
        code(EVAL_BLOCK, "nb02-eval"),
        code(MODEL_INFO_SUFFIX, "nb02-info"),
    ]
    return nb(cells)


# ─── notebook 03: VGG from scratch ───────────────────────────────────────────

def make_nb03():
    model_def = """\
def vgg_block(in_ch, out_ch, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGGScratch(nn.Module):
    \"\"\"VGG-style network trained from scratch on 128x128 inputs.\"\"\"
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,   64,  2),   # 128 -> 64
            vgg_block(64,  128, 2),   # 64  -> 32
            vgg_block(128, 256, 3),   # 32  -> 16
            vgg_block(256, 512, 3),   # 16  -> 8
            vgg_block(512, 512, 3),   # 8   -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 2048), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(2048, 512),         nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

def create_model(num_labels):
    return VGGScratch(num_classes=num_labels)

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 03 — VGG-style Network (from scratch)", "nb03-title"),
        code(COMMON_IMPORTS, "nb03-imports"),
        code(data_loading_block(128, 64, "vgg_scratch"), "nb03-data"),
        md("## Model Definition", "nb03-model-sec"),
        code(model_def, "nb03-model"),
        md("## Grid Search (LR × WD)", "nb03-gs-sec"),
        code(GRID_SEARCH, "nb03-grid"),
        md("## Final Training", "nb03-train-sec"),
        code(FINAL_TRAIN, "nb03-train"),
        md("## Evaluation", "nb03-eval-sec"),
        code(EVAL_BLOCK, "nb03-eval"),
        code(MODEL_INFO_SUFFIX, "nb03-info"),
    ]
    return nb(cells)


# ─── notebook 04: VGG pretrained ─────────────────────────────────────────────

def make_nb04():
    model_def = """\
from torchvision import models as tv_models

def create_model(num_labels):
    m = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(4096, num_labels)
    return m

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 04 — VGG-16 BN (pretrained ImageNet)", "nb04-title"),
        code(COMMON_IMPORTS, "nb04-imports"),
        code(data_loading_block(224, 32, "vgg16_pretrained"), "nb04-data"),
        md("## Model Definition", "nb04-model-sec"),
        code(model_def, "nb04-model"),
        md("## Grid Search (LR × WD)", "nb04-gs-sec"),
        code(GRID_SEARCH, "nb04-grid"),
        md("## Final Training", "nb04-train-sec"),
        code(FINAL_TRAIN, "nb04-train"),
        md("## Evaluation", "nb04-eval-sec"),
        code(EVAL_BLOCK, "nb04-eval"),
        code(MODEL_INFO_SUFFIX, "nb04-info"),
    ]
    return nb(cells)


# ─── notebook 05: ResNet-50 ───────────────────────────────────────────────────

def make_nb05():
    model_def = """\
from torchvision import models as tv_models

def create_model(num_labels):
    m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_labels)
    return m

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 05 — ResNet-50 (pretrained ImageNet)", "nb05-title"),
        code(COMMON_IMPORTS, "nb05-imports"),
        code(data_loading_block(224, 64, "resnet50_pretrained"), "nb05-data"),
        md("## Model Definition", "nb05-model-sec"),
        code(model_def, "nb05-model"),
        md("## Grid Search (LR × WD)", "nb05-gs-sec"),
        code(GRID_SEARCH, "nb05-grid"),
        md("## Final Training", "nb05-train-sec"),
        code(FINAL_TRAIN, "nb05-train"),
        md("## Evaluation", "nb05-eval-sec"),
        code(EVAL_BLOCK, "nb05-eval"),
        code(MODEL_INFO_SUFFIX, "nb05-info"),
    ]
    return nb(cells)


# ─── notebook 06: MobileNetV2 ─────────────────────────────────────────────────

def make_nb06():
    model_def = """\
from torchvision import models as tv_models

def create_model(num_labels):
    m = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V2)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_labels)
    return m

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 06 — MobileNet-V2 (pretrained ImageNet)", "nb06-title"),
        code(COMMON_IMPORTS, "nb06-imports"),
        code(data_loading_block(224, 64, "mobilenetv2_pretrained"), "nb06-data"),
        md("## Model Definition", "nb06-model-sec"),
        code(model_def, "nb06-model"),
        md("## Grid Search (LR × WD)", "nb06-gs-sec"),
        code(GRID_SEARCH, "nb06-grid"),
        md("## Final Training", "nb06-train-sec"),
        code(FINAL_TRAIN, "nb06-train"),
        md("## Evaluation", "nb06-eval-sec"),
        code(EVAL_BLOCK, "nb06-eval"),
        code(MODEL_INFO_SUFFIX, "nb06-info"),
    ]
    return nb(cells)


# ─── notebook 07: EfficientNet-B0 ────────────────────────────────────────────

def make_nb07():
    model_def = """\
from torchvision import models as tv_models

def create_model(num_labels):
    m = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_labels)
    return m

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 07 — EfficientNet-B0 (pretrained ImageNet)", "nb07-title"),
        code(COMMON_IMPORTS, "nb07-imports"),
        code(data_loading_block(224, 64, "efficientnet_b0_pretrained"), "nb07-data"),
        md("## Model Definition", "nb07-model-sec"),
        code(model_def, "nb07-model"),
        md("## Grid Search (LR × WD)", "nb07-gs-sec"),
        code(GRID_SEARCH, "nb07-grid"),
        md("## Final Training", "nb07-train-sec"),
        code(FINAL_TRAIN, "nb07-train"),
        md("## Evaluation", "nb07-eval-sec"),
        code(EVAL_BLOCK, "nb07-eval"),
        code(MODEL_INFO_SUFFIX, "nb07-info"),
    ]
    return nb(cells)


# ─── notebook 08: ViT (from scratch) ─────────────────────────────────────────

def make_nb08():
    model_def = """\
class VisionTransformer(nn.Module):
    \"\"\"Minimal ViT: 16x16 patches -> CLS token -> Transformer -> classifier.\"\"\"

    def __init__(self, num_classes, image_size=128, patch_size=16,
                 embed_dim=256, num_heads=8, depth=6, mlp_dim=512, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim        = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        B, C, H, W = x.shape
        ps      = self.patch_size
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.contiguous().view(B, C, -1, ps, ps)
        patches = patches.permute(0, 2, 1, 3, 4).flatten(2)
        x       = self.patch_embed(patches)
        cls     = self.cls_token.expand(B, -1, -1)
        x       = torch.cat((cls, x), dim=1)
        x       = self.dropout(x + self.pos_embed)
        x       = self.transformer(x)
        return self.head(self.norm(x[:, 0]))

def create_model(num_labels):
    return VisionTransformer(num_classes=num_labels, image_size=IMAGE_SIZE)

print_model_info(create_model(NUM_LABELS))
"""
    cells = [
        md("# 08 — Vision Transformer (from scratch)", "nb08-title"),
        code(COMMON_IMPORTS, "nb08-imports"),
        code(data_loading_block(128, 64, "vit_scratch"), "nb08-data"),
        md("## Model Definition", "nb08-model-sec"),
        code(model_def, "nb08-model"),
        md("## Grid Search (LR × WD)", "nb08-gs-sec"),
        code(GRID_SEARCH, "nb08-grid"),
        md("## Final Training", "nb08-train-sec"),
        code(FINAL_TRAIN, "nb08-train"),
        md("## Evaluation", "nb08-eval-sec"),
        code(EVAL_BLOCK, "nb08-eval"),
        code(MODEL_INFO_SUFFIX, "nb08-info"),
    ]
    return nb(cells)


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating notebooks...")
    save("01_data_and_baselines.ipynb", make_nb01())
    save("02_small_cnn.ipynb",          make_nb02())
    save("03_vgg_scratch.ipynb",        make_nb03())
    save("04_vgg_pretrained.ipynb",     make_nb04())
    save("05_resnet50.ipynb",           make_nb05())
    save("06_mobilenetv2.ipynb",        make_nb06())
    save("07_efficientnet_b0.ipynb",    make_nb07())
    save("08_vit.ipynb",                make_nb08())
    print("Done.")
