import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


CLASS_NAMES = ["not depressed", "depressed"]


def get_class_names(binning: dict | None) -> list[str]:
    if binning and binning.get("mode") == "phq_multiclass":
        bins = binning.get("bins", [])
        if bins:
            return [b["name"] for b in bins]
    return CLASS_NAMES


def _build_tags(config: dict | None) -> list[str]:
    base = ["fhad-daic", "depression-detection"]
    extra = (config or {}).get("tags", [])

    binning = (config or {}).get("binning", {})
    if binning.get("mode") == "phq_multiclass":
        n_bins = len(binning.get("bins", []))
        extra = extra + [
            "multiclass",
            f"{n_bins}-class",
            "phq-bins",
        ]

    return base + [t for t in extra if t not in base]


def _compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def get_bag_labels(y: torch.Tensor, sids: torch.Tensor) -> torch.Tensor:
    unique_sids, inverse = torch.unique(sids, sorted=True, return_inverse=True)
    n_bags = len(unique_sids)
    bag_y = y.new_zeros(n_bags, dtype=y.dtype)
    for b in range(n_bags):
        idx = (inverse == b).nonzero(as_tuple=True)[0][0]
        bag_y[b] = y[idx]
    return bag_y


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 0.0,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    grad_norms = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        gn = _compute_grad_norm(model)
        grad_norms.append(gn)
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(y)
    avg_loss = total_loss / len(loader.dataset)
    avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
    max_grad = float(np.max(grad_norms)) if grad_norms else 0.0
    return avg_loss, avg_grad, max_grad


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels


def train_epoch_mil(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 0.0,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_bags = 0
    grad_norms = []
    for X, y, sids in loader:
        X, y, sids = X.to(device), y.to(device), sids.to(device)
        optimizer.zero_grad()
        logits = model(X, sids)
        bag_y = get_bag_labels(y, sids)
        loss = criterion(logits, bag_y)
        loss.backward()
        gn = _compute_grad_norm(model)
        grad_norms.append(gn)
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(bag_y)
        total_bags += len(bag_y)
    avg_loss = total_loss / max(total_bags, 1)
    avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
    max_grad = float(np.max(grad_norms)) if grad_norms else 0.0
    return avg_loss, avg_grad, max_grad


def evaluate_mil(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y, sids in loader:
            X, y, sids = X.to(device), y.to(device), sids.to(device)
            logits = model(X, sids)
            bag_y = get_bag_labels(y, sids)
            total_loss += criterion(logits, bag_y).item() * len(bag_y)
            total_bags += len(bag_y)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(bag_y.cpu().tolist())
    avg_loss = total_loss / max(total_bags, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels


def _log_dev_metrics(epoch: int, all_preds: list[int], all_labels: list[int], class_names: list[str]) -> dict:
    metrics = {}

    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f"dev/f1_{name}"] = float(per_class_f1[i]) if len(per_class_f1) > i else 0.0
        metrics[f"dev/precision_{name}"] = float(per_class_precision[i]) if len(per_class_precision) > i else 0.0
        metrics[f"dev/recall_{name}"] = float(per_class_recall[i]) if len(per_class_recall) > i else 0.0

    if epoch > 0:
        metrics["dev/confusion_matrix"] = wandb.plot.confusion_matrix(
            y_true=all_labels, preds=all_preds, class_names=class_names,
            title="Confusion Matrix",
        )
        metrics["dev/pred_distribution"] = wandb.Histogram(all_preds)

    return metrics


def _gpu_memory_mb() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu/memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
    checkpoint_path: Path,
    device: torch.device,
    config: dict | None = None,
    is_mil: bool = False,
) -> dict:
    load_dotenv()

    class_names = get_class_names((config or {}).get("binning"))

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "fhad-tcn-depression"),
        entity=os.getenv("WANDB_ENTITY"),
        mode=os.getenv("WANDB_MODE", "online"),
        config=config,
        name=(config or {}).get("experiment_name", "fhad-daic-experiment"),
        tags=_build_tags(config),
    )

    wandb.watch(model, log="all", log_freq=100)

    scheduler = None
    if config and "scheduler" in config:
        s_cfg = config["scheduler"]
        if s_cfg.get("type") == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=s_cfg.get("mode", "min"),
                factor=s_cfg.get("factor", 0.5),
                patience=s_cfg.get("patience", 5),
                min_lr=s_cfg.get("min_lr", 1e-6),
            )

    best_f1 = 0.0
    epochs_without_improvement = 0
    history = []
    grad_clip = (config or {}).get("training", {}).get("grad_clip_norm", 0.0)

    bar = tqdm(range(1, num_epochs + 1), desc="Training")
    for epoch in bar:
        if is_mil:
            train_loss, train_grad_avg, train_grad_max = train_epoch_mil(
                model, train_loader, optimizer, criterion, device, max_grad_norm=grad_clip,
            )
            dev_loss, dev_f1, dev_preds, dev_labels = evaluate_mil(model, dev_loader, criterion, device)
        else:
            train_loss, train_grad_avg, train_grad_max = train_epoch(
                model, train_loader, optimizer, criterion, device, max_grad_norm=grad_clip,
            )
            dev_loss, dev_f1, dev_preds, dev_labels = evaluate(model, dev_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(dev_loss)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
            "dev_f1": dev_f1, "lr": current_lr,
        })

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/grad_norm_avg": train_grad_avg,
            "train/grad_norm_max": train_grad_max,
            "dev/loss": dev_loss,
            "dev/macro_f1": dev_f1,
            "lr": current_lr,
        }
        metrics.update(_log_dev_metrics(epoch, dev_preds, dev_labels, class_names))
        metrics.update(_gpu_memory_mb())
        wandb.log(metrics)

        bar.set_postfix_str(f"loss={train_loss:.3f}/{dev_loss:.3f} f1={dev_f1:.3f} lr={current_lr:.1e}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "dev_f1": dev_f1}, checkpoint_path)
            wandb.summary["best_dev_f1"] = best_f1
            wandb.summary["best_epoch"] = epoch
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    wandb.finish()
    return {"best_dev_f1": best_f1, "history": history}
