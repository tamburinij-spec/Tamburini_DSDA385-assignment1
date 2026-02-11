## Deep Learning Assignment 1 — Datasets × Architectures Benchmark

This repository implements **DSDA385 Assignment 1**:

- **3 datasets**: Adult Income (tabular), CIFAR-10 (images), PatchCamelyon / PCam (histopathology images)
- **3 architectures**: MLP, CNN, and attention-based (Transformer/ViT-style)
- **8 experiments total**: Adult uses MLP and CNN only (no attention); CIFAR-10 and PCam use all three architectures. Configured via YAML.

The **goals** are to compare how architectures interact with data modalities and to practice clean, reproducible deep learning experiments.

---

### Project structure

```
DSDA385-assigment-1/
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

### 1. Setup

- **Python**: 3.10+ (tested with 3.13)
- **Framework**: PyTorch

Install dependencies:

```bash
cd DSDA385-assigment-1
python -m venv .venv
.venv\Scripts\activate   # on Windows
# On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Datasets

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

Each dataset is wrapped in `data/datasets.py` so that all three expose **train/val/test** splits and `DataLoader`s with consistent interfaces.

---

### 3. Architectures

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

### 4. Training and Evaluation

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

### 5. Running the 8 Experiments

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

### 6. Results Table (fill after training)

After running all 8 experiments, each run creates a folder under `results/` named like `adult_mlp_20260210_111945`. Inside each folder, open **`final_metrics.json`** — it contains:

- **`accuracy`** — test accuracy (e.g. `0.8596` = 85.96%)
- **`f1`** — test F1 score (e.g. `0.6855` = 68.55%)
- **`loss`** — test loss (optional to report)

**How to fill the table:**

1. Go to `results/` and open each experiment folder (name pattern: `{dataset}_{architecture}_{date}_{time}`).
2. In each folder, open `final_metrics.json` and copy the `accuracy` and `f1` values.
3. Round to 2–4 decimal places (e.g. 85.96% or 0.8596) and paste into the table below.

**Test set results:**

| Dataset   | Architecture | Test Accuracy | Test F1 |
|-----------|--------------|----------------|---------|
| Adult     | MLP          | 85.97%         | 68.55%  |
| Adult     | CNN          | 85.78%         | 65.20%  |
| CIFAR-10  | MLP          | —              | —       |
| CIFAR-10  | CNN          | —              | —       |
| CIFAR-10  | Attention    | —              | —       |
| PCam      | MLP          | —              | —       |
| PCam      | CNN          | —              | —       |
| PCam      | Attention    | —              | —       |

*Adult uses MLP and CNN only. CIFAR-10 and PCam: run the corresponding configs and add values from each run’s `final_metrics.json`.*

**Compact view (accuracy only):**

|            | MLP    | CNN    | Attention |
|------------|--------|--------|------------|
| **Adult**  | 85.97% | 85.78% | —          |
| **CIFAR-10** | —    | —      | —          |
| **PCam**   | —      | —      | —          |

---

### 7. What to Analyze in Your Report

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

