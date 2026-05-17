# Multi-Label Object Classification

A machine learning project for **multi-label image classification** developed as part of **CAI6108**. Given a photograph that may contain several everyday objects at once, the goal is to predict which of twelve object categories are present—each label is predicted independently, so a single image can activate any subset of classes.

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

## Dataset

Images live under `data/aggregated/`. Each subdirectory name encodes the **set of labels** present in every image inside it, with labels joined by underscores (e.g. `book_chair_backpack/` contains scenes with a book, chair, and backpack). Individual files follow the naming pattern `img*.png`.

The aggregated dataset contains **~4,500 images** across hundreds of label combinations. A typical train/validation/test split uses a **70 / 15 / 15** ratio with a fixed random seed for reproducibility.

## Project Structure

```
MLE-Project/
├── eval.py                  # Official evaluation script (course submission interface)
├── utils.py                 # Plotting and dataset-splitting helpers
├── data/
│   └── aggregated/          # Multi-label image dataset
├── checkpoints/             # Saved model weights (gitignored)
├── experiments/             # Early exploratory notebooks
├── experiments2/            # Structured experiment pipeline (recommended)
│   ├── 01_Baseline.ipynb
│   ├── 02_CustomCNN_FromScratch.ipynb
│   ├── …
│   └── 08_FinalAnalysis.ipynb
└── final_experiments/       # Consolidated final-model notebooks
    ├── 01_data_and_baselines.ipynb
    ├── …
    ├── generate_figures.py  # Publication-quality EDA figures
    └── figures/             # Generated plots
```

## Models Explored

The project systematically compares architectures ranging from lightweight custom CNNs to modern transfer-learning backbones:

| Category | Architectures |
|----------|----------------|
| From scratch | Small CNN, VGG-16 |
| Classic transfer learning | ResNet-18/50, MobileNetV2, EfficientNet-B0 |
| Modern backbones | EfficientNetV2-S, ConvNeXt-Tiny |
| Transformers | Vision Transformer (ViT) |

Advanced techniques applied to the best-performing models include **Asymmetric Loss**, **RandAugment**, **stratified splitting**, **weighted random sampling**, and **per-class threshold tuning**. The final analysis notebook (`experiments2/08_FinalAnalysis.ipynb`) provides head-to-head comparisons, confusion matrices, GradCAM visualisations, and error analysis.

## Requirements

- **Python** ≥ 3.13
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)
- **CUDA** optional but recommended for training (PyTorch wheels target CUDA 12.8)

Core dependencies: PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, seaborn, Jupyter.

## Setup

Clone the repository and install dependencies with `uv`:

```bash
git clone <repository-url>
cd MLE-Project
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate   # Linux / macOS
```

## Usage

### Running Experiments

Open and run the notebooks in order. The `experiments2/` directory is the primary, narrative-driven pipeline:

1. **01_Baseline** — dataset exploration and non-learned baselines
2. **02–06** — progressively more capable architectures
3. **07_BestModel_Improvements** — targeted engineering on the top backbone
4. **08_FinalAnalysis** — consolidated evaluation and interpretability

The `final_experiments/` notebooks reproduce the key models in a streamlined form suitable for report figures.

Generate exploratory figures for the write-up:

```bash
python final_experiments/generate_figures.py
```

Output is saved to `final_experiments/figures/`.

### Official Evaluation

`eval.py` is the course-provided evaluation entry point. It loads a checkpoint, runs inference on a test directory, and reports standardised metrics. Update `load_trained_model()` and `predict()` so they match your saved architecture before submission.

```bash
python eval.py \
  --model_path checkpoints/final_resnet50_pretrained.pth \
  --test_data project_test_data \
  --group_id <YOUR_GROUP_ID> \
  --project_title "Your Project Title"
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

## Reproducibility

- Random seed **42** is used consistently across notebooks for splits, sampling, and weight initialisation.
- Image preprocessing follows ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Checkpoints are stored in `checkpoints/` and excluded from version control.

## License

This repository was developed for academic coursework. Contact the authors before reusing code or data outside the course context.
