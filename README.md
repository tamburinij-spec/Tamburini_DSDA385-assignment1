## Deep Learning Assignment 1 — Datasets × Architectures Benchmark

This repository implements **DSDA385 Assignment 1**:

- **3 datasets**: Adult Income (tabular), CIFAR-10 (images), PatchCamelyon / PCam (histopathology images)
- **3 architectures**: MLP, CNN, and attention-based (Transformer/ViT-style)
- **8 experiments total**: Adult uses MLP and CNN only (no attention); CIFAR-10 and PCam use all three architectures. Configured via YAML.

The **goals** are to compare how architectures interact with data modalities and to practice clean, reproducible deep learning experiments.

---

### Project structure

```
DSDA385-assignment-1/
├── configs/           # YAML configs (Adult: mlp, cnn; CIFAR-10/PCam: mlp, cnn, attention)
├── data/              # Dataset loading and preprocessing
│   ├── datasets.py    # create_dataloaders() for Adult, CIFAR-10, PCam
│   └── cifar10/       # CIFAR-10 data (downloaded on first run)
├── models/            # Model definitions
│   ├── factory.py     # create_model() — dispatches by architecture and dataset
│   ├── mlp.py         # TabularMLP, ImageMLP
│   ├── cnn.py         # SimpleCNN
│   └── attention.py   # TabularTransformer, VisionTransformerTiny
├── utils/             # Config loading and training loop
│   ├── config.py      # load_config()
│   └── training.py    # train_one_experiment(), early stopping, plotting
├── results/           # Timestamped run outputs (metrics, curves, config)
├── train.py           # Main entry point
├── requirements.txt   # Dependencies
└── README.md
```

---

### Setup

- **Python**: 3.10+ (tested with 3.13)
- **Framework**: PyTorch

Install dependencies:

```bash
cd DSDA385-assignment-1
python -m venv .venv
.venv\Scripts\activate   # on Windows
```

---

### Datasets

- **Adult (UCI Adult Income, tabular)**
  - Loaded via `sklearn.fetch_openml("adult", version=2, as_frame=True)`
  - Mixed numerical + categorical features
  - Binary label: income \> 50K
  - Metrics: accuracy, F1

- **CIFAR-10 (natural images)**
  - `torchvision.datasets.CIFAR10`
  - 32×32 RGB color images
  - 10 classes (0–9)
  - Data augmentation: random crop + horizontal flip on training

- **PCam (PatchCamelyon, histopathology)**
  - `torchvision.datasets.PCAM`
  - 96×96 RGB patches from H&E-stained slides
  - Binary label: normal vs tumor tissue
  - Metric: accuracy, F1
  - **If automatic download fails** with *"Too many users have viewed or downloaded this file recently"* (Google Drive quota), either **wait a few hours and run again**, or **download manually**: create `data/pcam/pcam/`, download the 6 `.gz` files from the links below, put them in that folder, then decompress each (e.g. 7-Zip, or `gzip -d *.gz`) so you have the `.h5` files. Then run training with `download=True` (it will skip download if the `.h5` files exist).
    - [train images](https://drive.google.com/uc?id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2) → `camelyonpatch_level_2_split_train_x.h5.gz`
    - [train targets](https://drive.google.com/uc?id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG) → `camelyonpatch_level_2_split_train_y.h5.gz`
    - [val images](https://drive.google.com/uc?id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3) → `camelyonpatch_level_2_split_valid_x.h5.gz`
    - [val targets](https://drive.google.com/uc?id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO) → `camelyonpatch_level_2_split_valid_y.h5.gz`
    - [test images](https://drive.google.com/uc?id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_) → `camelyonpatch_level_2_split_test_x.h5.gz`
    - [test targets](https://drive.google.com/uc?id=17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP) → `camelyonpatch_level_2_split_test_y.h5.gz`
    - After downloading, rename each file to the name shown above if needed. Then decompress (e.g. 7-Zip, or run `python data/decompress_pcam.py` from the project root) so the `.h5` files are in `data/pcam/pcam/`.

Each dataset is wrapped in `data/datasets.py` so that all three expose **train/val/test** splits and `DataLoader`s with consistent interfaces.

---

### Architectures

- **MLP (Multilayer Perceptron)**
  - `models/mlp.py`
  - For **tabular**: `TabularMLP` (2+ hidden layers, ReLU, BatchNorm, dropout)
  - For **images**: `ImageMLP` that flattens the image and applies fully-connected layers

- **CNN (Convolutional Neural Network)**
  - `models/cnn.py` — `SimpleCNN`
  - At least 2 convolutional layers with pooling, BatchNorm, ReLU, and a fully-connected head
  - Works for CIFAR-10 and PCam (different input resolutions handled by adaptive pooling)

- **Attention-Based / Deep Feature Model**
  - `models/attention.py`
  - **Tabular**: `TabularTransformer` — Transformer encoder operating over feature tokens (not used for Adult in this assignment)
  - **Images**: `VisionTransformerTiny` — small ViT-style model with patch embeddings, positional encodings, and Transformer encoder layers; used for CIFAR-10 and PCam

All architectures are created via the factory in `models/factory.py`. For this assignment, **Adult** is run with MLP and CNN only; **CIFAR-10** and **PCam** use all three (MLP, CNN, attention).

---

### Training and Evaluation

The main entry point is `train.py`, which takes a **YAML config** describing one experiment:

```bash
python train.py --config configs/adult_mlp.yaml
```

Key config sections:

- **`dataset`**: name (`adult`, `cifar10`, `pcam`), batch size, num workers, etc.
- **`model`**: architecture (`mlp`, `cnn`, `attention`) and model hyperparameters.
- **`training`**: epochs, learning rate, weight decay, early stopping config.

For each run, the script:

- Creates a timestamped folder under `results/EXPERIMENT_NAME_YYYYMMDD_HHMMSS/`
- Saves:
  - `config_resolved.json` — the exact config used
  - `history.json` — training/validation loss & accuracy per epoch
  - `final_metrics.json` — final test accuracy and F1
  - **`loss_curve.png`** — learning curve (train vs validation loss per epoch)
  - **`accuracy_curve.png`** — learning curve (train vs validation accuracy per epoch)

**Plots in the results folder**

Each experiment folder contains two PNG plots:

| File | Description |
|------|-------------|
| **loss_curve.png** | Train and validation loss vs epoch. Use it to check convergence and overfitting (e.g. val loss rising while train loss falls). |
| **accuracy_curve.png** | Train and validation accuracy vs epoch. Use it to see how quickly the model learns and whether validation accuracy plateaus or drops. |

Both plots show two lines (train and val). You can use them in your report to compare stability and overfitting across architectures or datasets.

**Why early stopping?**

Training uses **early stopping** so that we stop once the model stops improving on the validation set, instead of always running for the full number of epochs. This:

- **Reduces overfitting** — we keep the model from the epoch with the best validation loss, not the last epoch (which may have started overfitting).
- **Saves time** — we don’t waste epochs when validation loss has already plateaued or started to rise.

The logic in `utils/training.py` monitors **validation loss** each epoch. If validation loss does not improve for `patience` consecutive epochs (e.g. 5), training stops and the **best model weights** (by validation loss) are restored before the final test evaluation. So reported test metrics are for the best checkpoint, not the last epoch.

Early stopping is controlled in each experiment’s config under `training.early_stopping`:

- **`enabled`** — turn early stopping on or off (default: true).
- **`patience`** — number of epochs without improvement before stopping (default: 5).
- **`min_delta`** — minimum change in validation loss to count as improvement (default: 0.0).

---

### Running the 8 Experiments

Configs are under `configs/`. **Adult** uses only MLP and CNN (no attention); **CIFAR-10** and **PCam** use all three architectures.

- **Adult** (MLP, CNN only)
  - `configs/adult_mlp.yaml`
  - `configs/adult_cnn.yaml`

- **CIFAR-10**
  - `configs/cifar10_mlp.yaml`
  - `configs/cifar10_cnn.yaml`
  - `configs/cifar10_attention.yaml`

- **PCam**
  - `configs/pcam_mlp.yaml`
  - `configs/pcam_cnn.yaml`
  - `configs/pcam_attention.yaml`

Example commands:

```bash
python train.py --config configs/adult_mlp.yaml
python train.py --config configs/adult_cnn.yaml

python train.py --config configs/cifar10_mlp.yaml
python train.py --config configs/cifar10_cnn.yaml
python train.py --config configs/cifar10_attention.yaml

python train.py --config configs/pcam_mlp.yaml
python train.py --config configs/pcam_cnn.yaml
python train.py --config configs/pcam_attention.yaml
```

---

### Results Table (filled after training)

After running all 8 experiments, each run creates a folder under `results/` named like `adult_mlp_20260210_111945`. Inside each folder, open **`final_metrics.json`** — it contains:

- **`accuracy`** — test accuracy (e.g. `0.8596` = 85.96%)
- **`f1`** — test F1 score (e.g. `0.6855` = 68.55%)
- **`loss`** — test loss (optional to report)

**Test set results:**

| Dataset   | Architecture | Test Accuracy  | Test F1 |
|-----------|--------------|----------------|---------|
| Adult     | MLP          | 85.97%         | 68.55%  |
| Adult     | CNN          | 85.78%         | 65.20%  |
| CIFAR-10  | MLP          | 37.45%         | 36.28%  |
| CIFAR-10  | CNN          | 81.75%         | 81.46%  |
| CIFAR-10  | Attention    | 74.62%         | 74.40%  |
| PCam      | MLP          | 55.84%         | 27.98%  |
| PCam      | CNN          | 82.66%         | 81.08%  |
| PCam      | Attention    | 79.88%         | 78.91%  |

*Adult uses MLP and CNN only. PCam: run the corresponding configs and add values from each run’s `final_metrics.json`.*

---

### What to Analyze in Your Report

These are suggestions you can adapt when you write `README`/report text for submission:

- **Dataset characteristics**
  - Adult: mixed numeric + categorical; relatively low-dimensional; we use MLP and CNN only (no attention).
  - CIFAR-10: small natural images; strong spatial locality; CNNs and ViT-style models should outperform MLP.
  - PCam: medical images; subtle texture cues; CNNs and ViTs can capture local and global patterns.

- **Architecture behavior**
  - How does the **MLP** compare to the **CNN** on image datasets?
  - Does the **attention-based** model help on CIFAR-10 and PCam compared to MLP/CNN?
  - Is there overfitting? Check learning curves and F1 vs accuracy.

- **Takeaways**
  - Which inductive bias (fully-connected vs convolution vs attention) is best suited to each modality?
  - How does training stability and convergence speed differ across models?

Use the saved metrics and plots under `results/` to support your explanations with evidence.

---

### Objective

This assignment benchmarks how different neural architectures behave across different data modalities:

- Tabular data → Adult
- Natural images → CIFAR-10
- Medical images → PCam

The goal is to understand how **model inductive bias** interacts with **data structure**, not just to maximize accuracy.

---

### Code Design

The project is fully modular and configuration-driven.

- **Configs (`configs/`)**
  - YAML files define dataset, architecture, optimizer, epochs, batch size, early stopping.
  - No code modification required to change experiments.

- **Model Factory (`models/factory.py`)**
  - Dynamically creates models (`mlp`, `cnn`, `attention`).
  - Ensures clean separation between architectures.

- **Training Pipeline (`utils/training.py`)**
  - Shared training loop for all experiments.
  - Early stopping based on validation loss.
  - Saves:
    - `history.json`
    - `final_metrics.json`
    - `loss_curve.png`
    - `accuracy_curve.png`

All experiments use:
- Adam optimizer
- Consistent train/val/test splits
- No pretrained models
- ≤ 1 hour per model

---

### Dataset Insights

#### Adult (Tabular)

- Mixed numerical + categorical features
- Binary classification (>50K)

**Best model:** MLP  
Reason: Tabular data has no spatial structure; fully connected layers are sufficient.

---

#### CIFAR-10 (Natural Images)

- 32×32 RGB
- 10 classes

**Best model:** CNN  
Reason:
- CNN preserves spatial locality.
- MLP destroys structure when flattening.
- Attention works but is less sample-efficient than CNN on small datasets.

---

#### PCam (Histopathology)

- 96×96 RGB
- Binary tumor detection
- Texture-based patterns

**Best model:** CNN  
Reason:
- CNN captures local texture features.
- Attention competitive but slightly lower.
- MLP fails due to loss of spatial inductive bias.

---

### Architecture Comparison

| Architecture | Strength | Weakness |
|--------------|----------|----------|
| MLP | Works well for structured/tabular data | Poor on images (no spatial bias) |
| CNN | Strong spatial inductive bias | Limited global context |
| Attention | Captures global relationships | Requires more data, less sample-efficient |

---

### Learning Curve Observations

- MLP on images → overfitting and poor generalization.
- CNN → stable convergence and best validation performance.
- Attention → slower convergence but competitive.
- Early stopping prevented over-training.

---

### Key Takeaways

1. Inductive bias must match data modality.
2. CNN is optimal for image-based tasks.
3. MLP is sufficient for tabular data.
4. Attention models are powerful but data-hungry.
5. Clean modular design enables reproducible experiments.

---

### Reproducibility

To reproduce any experiment:

```bash
python train.py --config configs/<experiment>.yaml
```

---

### Learning Curve Comparison

Learning curves were analyzed using `history.json` for each experiment.

### Convergence Speed

- **Adult (MLP vs CNN)**
  - Both converge quickly (<10 epochs).
  - Very small generalization gap.
  - Stable optimization (tabular data is low-dimensional).

- **CIFAR-10**
  - MLP plateaus early (~38% val acc).
  - CNN steadily improves to ~81.8%.
  - Attention improves more gradually but consistently to ~74.6%.
  - CNN shows faster and more stable convergence.

- **PCam**
  - MLP unstable after few epochs (validation collapse).
  - CNN steadily improves to ~82.7%.
  - Attention competitive (~79.9%) but with more variance.
  - CNN demonstrates strongest stability.

---

## Parameter Efficiency vs Performance

Although exact parameter counts vary:

- **MLP (Images)**
  - Very large parameter count (due to flattening 32×32×3 or 96×96×3).
  - Poor accuracy.
  - Inefficient use of parameters.

- **CNN**
  - Weight sharing drastically reduces parameters.
  - Highest performance on image datasets.
  - Most parameter-efficient architecture.

- **Attention (ViT-style)**
  - Higher parameter complexity than CNN.
  - Competitive but not superior.
  - Requires larger datasets to outperform CNN.

Key Insight:
Higher parameter count ≠ better performance.
Architectural inductive bias matters more than raw capacity.

---

## Architecture Efficiency Summary

| Dataset   | Most Efficient Model | Reason |
|------------|---------------------|--------|
| Adult      | MLP                 | Matches tabular structure |
| CIFAR-10   | CNN                 | Spatial inductive bias |
| PCam       | CNN                 | Texture feature extraction |

CNN achieves the best trade-off between:
- Accuracy
- Stability
- Parameter efficiency
- Convergence speed

---

## Overall Experimental Insight

1. Matching architecture to modality is critical.
2. CNN remains the strongest baseline for vision tasks.
3. Attention models are competitive but require scale.
4. MLP scales poorly for high-dimensional image inputs.
5. Proper early stopping prevents severe overfitting (especially on PCam).

This bonus analysis reinforces that model inductive bias dominates performance more than parameter count alone.