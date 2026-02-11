from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0


def _get_early_stopping(cfg: Dict[str, Any]) -> EarlyStoppingConfig:
    es_cfg = cfg.get("early_stopping", {}) or {}
    return EarlyStoppingConfig(
        enabled=bool(es_cfg.get("enabled", True)),
        patience=int(es_cfg.get("patience", 5)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool,
) -> Tuple[float, float]:
    model.train()
    all_preds = []
    all_targets = []
    running_loss = 0.0

    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # For classification (including binary with 2 logits), always use
        # CrossEntropyLoss with integer class targets.
        loss = criterion(outputs, targets)
        preds = outputs.argmax(dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_preds_tensor = torch.cat(all_preds)
    all_targets_tensor = torch.cat(all_targets)

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets_tensor.numpy(), all_preds_tensor.numpy())
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Same as training: CrossEntropy over logits.
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds_tensor = torch.cat(all_preds)
    all_targets_tensor = torch.cat(all_targets)

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets_tensor.numpy(), all_preds_tensor.numpy())
    if is_binary:
        # Binary F1 for two-class problems (labels 0/1)
        f1 = f1_score(all_targets_tensor.numpy(), all_preds_tensor.numpy(), average="binary")
    else:
        f1 = f1_score(all_targets_tensor.numpy(), all_preds_tensor.numpy(), average="macro")

    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


def plot_curves(history: Dict[str, Any], out_dir: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()


def train_one_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_cfg: Dict[str, Any],
    exp_dir: str,
    is_binary: bool,
) -> Dict[str, Any]:
    num_epochs: int = int(train_cfg.get("num_epochs", 20))
    es = _get_early_stopping(train_cfg)

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "final_metrics": {},
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, is_binary
        )
        val_metrics = evaluate(model, val_loader, criterion, device, is_binary)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f}"
        )

        # Early stopping check
        if val_metrics["loss"] + es.min_delta < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if es.enabled and epochs_no_improve >= es.patience:
                print("Early stopping triggered.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, criterion, device, is_binary)
    history["final_metrics"] = test_metrics

    # Save metrics and curves
    with open(os.path.join(exp_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, exp_dir)

    return history

