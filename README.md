# CAI6108 — Multi-Label Object Classification (Group 10)

A machine learning project for **multi-label image classification** developed as part of **CAI6108: Machine Learning Engineering** at the **University of Florida**. Given a photograph that may contain several everyday objects at once, the goal is to predict which of twelve object categories are present—each label is predicted independently, so a single image can activate any subset of classes.

**Final model:** ResNet-50 (ImageNet pre-trained, fully fine-tuned) with Asymmetric Loss, Weighted Sampling, Cosine Annealing, per-class threshold tuning, and Test-Time Augmentation (TTA).

---

## Problem Overview

Unlike standard single-label classification, each image is associated with a **12-dimensional binary vector**: one entry per object category, set to `1` if that object appears in the scene and `0` otherwise. Models are trained with sigmoid outputs and evaluated using multi-label metrics (exact match, Hamming accuracy, mean IoU, micro-averaged precision/recall/F1).

### Object Categories

| # | Label | # | Label |
|---|-------|---|-------|
| 1 | pen | 7 | desk |
| 2 | paper | 8 | bottle |
| 3 | book | 9 | keychain |
| 4 | clock | 10 | backpack |
| 5 | phone | 11 | calculator |
| 6 | laptop | 12 | chair |

---

## Dataset

Images live under `data/aggregated/`. Each subdirectory name encodes the **set of labels** present in every image inside it, with labels joined by underscores (e.g. `book_chair_backpack/` contains scenes with a book, chair, and backpack). Individual files follow the naming pattern `img*.png`.

The aggregated dataset contains **~4,500 images** across hundreds of label combinations. A typical train/validation/test split uses a **70 / 15 / 15** ratio with a fixed random seed for reproducibility.

---

## Repository Structure

```
MLE-Project/
├── eval.py                        # Official evaluation script (course submission interface)
├── utils.py                       # Shared utilities (plotting, dataset splitting)
├── best_model.pth                 # Symlink → checkpoints/final_resnet50_scratch.pth
├── pyproject.toml                 # Project dependencies (managed with uv)
├── MLE_Report_Group_10.pdf        # Final project report
│
├── final_experiments/             # All experiment notebooks
│   ├── 01_data_and_baselines.ipynb
│   ├── 02_small_cnn.ipynb
│   ├── 03_vgg_scratch.ipynb
│   ├── 04_vgg_pretrained.ipynb
│   ├── 05_resnet50.ipynb
│   ├── 06_mobilenetv2.ipynb
│   ├── 07_efficientnet_b0.ipynb
│   ├── 08_vit.ipynb
│   ├── 09_finetuning.ipynb        # Transfer learning + ASL + TTA + threshold tuning
│   ├── 10_end_to_end_finetuning.ipynb  # Final model — end-to-end training (best results)
│   ├── generate_figures.py        # Publication-quality EDA figures
│   ├── generate_notebooks.py
│   └── figures/                   # Generated plots
│
├── checkpoints/                   # Saved model weights (.pth files)
│
└── data/
    └── aggregated/                # Dataset — one sub-folder per label combination
```

---

## Models Explored

The project systematically compares architectures ranging from lightweight custom CNNs to modern transfer-learning backbones:

| Category | Architectures |
|----------|---------------|
| From scratch | Small CNN, VGG-16, ViT |
| Classic transfer learning | ResNet-50, MobileNetV2, EfficientNet-B0 |
| VGG with pre-training | VGG (ImageNet weights) |
| Final (best) | ResNet-50 fully fine-tuned + ASL + TTA + threshold tuning |

---

## Final Model

**Architecture:** ResNet-50 (ImageNet pre-trained weights, fully fine-tuned)  
**Checkpoint:** `checkpoints/final_resnet50_scratch.pth` (also accessible via `best_model.pth`)

| Training technique | Detail |
|--------------------|--------|
| Loss | AsymmetricLoss (γ⁻=4, γ⁺=1, clip=0.05) |
| Sampler | WeightedRandomSampler (up-samples rare label combos) |
| Optimizer | AdamW (backbone lr=1e-4, head lr=5e-4, wd=1e-4) |
| LR schedule | Cosine annealing with 3-epoch linear warm-up |
| Precision | Mixed precision (torch.cuda.amp) |
| TTA | 8 augmented views averaged at inference |
| Thresholds | Per-class threshold tuning on validation set |
| Early stopping | 10 epochs patience on val micro-F1 |

See [final_experiments/10_end_to_end_finetuning.ipynb](final_experiments/10_end_to_end_finetuning.ipynb) for the full training pipeline.

---

## Setup

### Requirements

- **Python** ≥ 3.13
- **CUDA** 12.8 (optional but recommended for training; CPU also works)

### Install with uv (recommended)

```bash
pip install uv          # if not already installed
git clone <repository-url>
cd MLE-Project
uv sync
source .venv/bin/activate   # Linux / macOS
```

### Install with pip

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas matplotlib scikit-learn scipy seaborn jupyter h5py imutils
```

---

## Usage

### Running Experiments

Open and run the notebooks in `final_experiments/` in order:

| Notebook | Description |
|----------|-------------|
| `01_data_and_baselines.ipynb` | Dataset exploration, label statistics, frequency baselines |
| `02_small_cnn.ipynb` | Small custom CNN from scratch |
| `03_vgg_scratch.ipynb` | VGG-style network from scratch |
| `04_vgg_pretrained.ipynb` | VGG with ImageNet pre-training |
| `05_resnet50.ipynb` | ResNet-50 transfer learning |
| `06_mobilenetv2.ipynb` | MobileNetV2 transfer learning |
| `07_efficientnet_b0.ipynb` | EfficientNet-B0 transfer learning |
| `08_vit.ipynb` | Vision Transformer (ViT) from scratch |
| `09_finetuning.ipynb` | ResNet-50 + ASL + TTA + per-class threshold tuning |
| `10_end_to_end_finetuning.ipynb` | **Final model** — end-to-end training with all techniques |

Generate exploratory figures for the write-up:

```bash
python final_experiments/generate_figures.py
```

Output is saved to `final_experiments/figures/`.

### Official Evaluation

`eval.py` is the course-provided evaluation entry point. It loads a checkpoint, runs inference on a test directory, and reports standardised metrics.

```bash
python eval.py \
  --model_path best_model.pth \
  --test_data path/to/test_data \
  --group_id 10 \
  --project_title "Multi-Label Object Recognition" \
  --image_size 224
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to `.pth` checkpoint | *(required)* |
| `--test_data` | Directory with class subfolders | `project_test_data` |
| `--group_id` | Non-negative group identifier | *(required)* |
| `--project_title` | Project title (≥ 4 characters) | *(required)* |
| `--batch_size` | Inference batch size | `32` |
| `--image_size` | Input resolution (pixels) | `128` |
| `--threshold` | Sigmoid decision threshold | `0.5` |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Binary cross-entropy with logits |
| `exact_match` | Fraction of samples where every label is correct |
| `hamming_acc` | Per-label accuracy averaged over all labels |
| `mean_iou` | Mean Jaccard index across samples |
| `precision_micro` | Micro-averaged precision |
| `recall_micro` | Micro-averaged recall |
| `f1_micro` | Micro-averaged F1 score |

---

## Reproducibility

- Random seed **42** is used consistently across notebooks for splits, sampling, and weight initialisation.
- Image preprocessing follows ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Checkpoints are stored in `checkpoints/` and excluded from version control.

---

## License

This project was developed for **CAI6108: Machine Learning Engineering** at the **University of Florida**. The University of Florida and respective collaborators own all rights in this repository, including source code, models, and data. See [LICENSE](LICENSE) for terms of use. Contact the course instructional staff or UF for permissions beyond authorized course use.
