import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.config import load_config
from data.datasets import create_dataloaders
from models.factory import create_model
from utils.training import train_one_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Learning Assignment 1 runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file describing one experiment.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to store metrics, checkpoints and plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu or cuda). If not set, auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    experiment_name = config.get("experiment_name") or os.path.splitext(
        os.path.basename(args.config)
    )[0]

    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.join(
        args.output_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    # Save resolved config for reproducibility
    with open(os.path.join(exp_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Device
    if args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
        config["dataset"]
    )

    # Model
    model: nn.Module = create_model(
        architecture=config["model"]["architecture"],
        input_dim=input_dim,
        num_classes=num_classes,
        dataset_type=config["dataset"]["name"],
        model_cfg=config["model"],
    ).to(device)

    # Optimizer & loss
    train_cfg: Dict[str, Any] = config["training"]
    lr: float = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay: float = float(train_cfg.get("weight_decay", 0.0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # For this assignment all tasks are classification (including binary),
    # so we always use CrossEntropyLoss with integer class labels.
    criterion = nn.CrossEntropyLoss()

    history = train_one_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_cfg=train_cfg,
        exp_dir=exp_dir,
        is_binary=(num_classes == 2),
    )

    # Save final metrics separately
    with open(os.path.join(exp_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history["final_metrics"], f, indent=2)

    print("Experiment complete. Results stored in:", exp_dir)


if __name__ == "__main__":
    main()

