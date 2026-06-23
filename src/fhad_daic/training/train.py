import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_checkpoint


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

    auto = []

    binning = (config or {}).get("binning", {})
    if binning.get("mode") == "phq_multiclass":
        n_bins = len(binning.get("bins", []))
        auto.extend(["multiclass", f"{n_bins}-class", "phq-bins"])

    modality = (config or {}).get("features", {}).get("modality", "au")
    if modality != "au":
        auto.append(f"modality-{modality}")

    merged = auto + [t for t in extra if t not in auto]
    return base + [t for t in merged if t not in base]


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
) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            probs = torch.nan_to_num(torch.softmax(logits, dim=1).cpu(), nan=0.0)
            probs = torch.nan_to_num(probs, nan=0.0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = _compute_auc(all_labels, all_probs)
    return avg_loss, macro_f1, auc, all_preds, all_labels

def _compute_auc(labels: list[int], probs: list[list[float]]) -> float:
    labels = np.array(labels)
    probs = np.array(probs)
    row_sums = probs.sum(axis=1, keepdims=True)
    zero = (row_sums.squeeze() == 0)
    if zero.any():
        probs[zero] = 1.0 / probs.shape[1]
    non_zero = ~zero
    if non_zero.any():
        probs[non_zero] = probs[non_zero] / row_sums[non_zero]
    mask = ~np.isnan(probs).any(axis=1)
    if mask.sum() < 2:
        return 0.0
    labels = labels[mask]
    probs = probs[mask]
    n_classes = probs.shape[1]
    if n_classes == 2:
        return float(roc_auc_score(labels, probs[:, 1]))
    try:
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return 0.0


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
) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_bags = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y, sids in loader:
            X, y, sids = X.to(device), y.to(device), sids.to(device)
            logits = model(X, sids)
            bag_y = get_bag_labels(y, sids)
            total_loss += criterion(logits, bag_y).item() * len(bag_y)
            total_bags += len(bag_y)
            probs = torch.nan_to_num(torch.softmax(logits, dim=1).cpu(), nan=0.0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(bag_y.cpu().tolist())
            all_probs.extend(probs.tolist())
    avg_loss = total_loss / max(total_bags, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = _compute_auc(all_labels, all_probs)
    return avg_loss, macro_f1, auc, all_preds, all_labels


def train_epoch_fusion(
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
    for F_v, F_a, y, aux_v, aux_a in loader:
        F_v = F_v.to(device)
        F_a = F_a.to(device)
        y = y.to(device)
        aux_v = {k: v.to(device) for k, v in aux_v.items()}
        aux_a = {k: v.to(device) for k, v in aux_a.items()}
        optimizer.zero_grad()
        loss = criterion(model(F_v, F_a, aux_v, aux_a), y)
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


def train_epoch_fusion_tcn(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 0.0,
    grad_accum_steps: int = 1,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    grad_norms = []
    optimizer.zero_grad()
    for i, (X_v, mask_v, X_a, mask_a, y, aux_v, aux_a) in enumerate(loader):
        X_v = X_v.to(device)
        mask_v = mask_v.to(device)
        X_a = X_a.to(device)
        mask_a = mask_a.to(device)
        y = y.to(device)
        aux_v = {k: v.to(device) for k, v in aux_v.items()}
        aux_a = {k: v.to(device) for k, v in aux_a.items()}
        loss = criterion(model(X_v, mask_v, X_a, mask_a, aux_v, aux_a), y) / grad_accum_steps
        loss.backward()
        total_loss += loss.item() * len(y) * grad_accum_steps
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader.dataset):
            gn = _compute_grad_norm(model)
            grad_norms.append(gn)
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
    avg_loss = total_loss / len(loader.dataset)
    avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
    max_grad = float(np.max(grad_norms)) if grad_norms else 0.0
    return avg_loss, avg_grad, max_grad


def evaluate_fusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for F_v, F_a, y, aux_v, aux_a in loader:
            F_v = F_v.to(device)
            F_a = F_a.to(device)
            y = y.to(device)
            aux_v = {k: v.to(device) for k, v in aux_v.items()}
            aux_a = {k: v.to(device) for k, v in aux_a.items()}
            logits = model(F_v, F_a, aux_v, aux_a)
            total_loss += criterion(logits, y).item() * len(y)
            probs = torch.nan_to_num(torch.softmax(logits, dim=1).cpu(), nan=0.0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = _compute_auc(all_labels, all_probs)
    return avg_loss, macro_f1, auc, all_preds, all_labels


def evaluate_fusion_tcn(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_v, mask_v, X_a, mask_a, y, aux_v, aux_a in loader:
            X_v = X_v.to(device)
            mask_v = mask_v.to(device)
            X_a = X_a.to(device)
            mask_a = mask_a.to(device)
            y = y.to(device)
            aux_v = {k: v.to(device) for k, v in aux_v.items()}
            aux_a = {k: v.to(device) for k, v in aux_a.items()}
            logits = model(X_v, mask_v, X_a, mask_a, aux_v, aux_a)
            total_loss += criterion(logits, y).item() * len(y)
            probs = torch.nan_to_num(torch.softmax(logits, dim=1).cpu(), nan=0.0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = _compute_auc(all_labels, all_probs)
    return avg_loss, macro_f1, auc, all_preds, all_labels


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
    checkpoint_root: Path,
    device: torch.device,
    config: dict | None = None,
    is_mil: bool = False,
    is_fusion: bool = False,
    is_fusion_tcn: bool = False,
) -> dict:
    load_dotenv()

    class_names = get_class_names((config or {}).get("binning"))

    experiment_name = (config or {}).get("experiment_name", "experiment")
    experiment_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment_name)

    sweep_name = (config or {}).get("output", {}).get("sweep_name", "")
    sweep_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in sweep_name) if sweep_name else ""

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if sweep_name:
        sweep_dir = checkpoint_root / sweep_name
        config_dir = sweep_dir / experiment_name
    else:
        sweep_dir = None
        config_dir = checkpoint_root / experiment_name
    run_dir = config_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-config latest symlink
    config_latest = config_dir / "latest"
    if config_latest.is_symlink() or config_latest.exists():
        config_latest.unlink()
    config_latest.symlink_to(run_timestamp, target_is_directory=True)

    # Per-sweep latest (if inside a sweep)
    if sweep_dir:
        sweep_latest = sweep_dir / "latest"
        if sweep_latest.is_symlink() or sweep_latest.exists():
            sweep_latest.unlink()
        sweep_latest.symlink_to(Path(experiment_name) / run_timestamp, target_is_directory=True)

    # Global latest
    global_rel = Path(sweep_name) / experiment_name / run_timestamp if sweep_name else Path(experiment_name) / run_timestamp
    global_latest = checkpoint_root / "latest"
    if global_latest.is_symlink() or global_latest.exists():
        global_latest.unlink()
    global_latest.symlink_to(global_rel, target_is_directory=True)

    output_cfg = (config or {}).get("output", {})
    save_frequency = output_cfg.get("save_frequency", 0)

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
            dev_loss, dev_f1, dev_auc, dev_preds, dev_labels = evaluate_mil(model, dev_loader, criterion, device)
        elif is_fusion:
            train_loss, train_grad_avg, train_grad_max = train_epoch_fusion(
                model, train_loader, optimizer, criterion, device, max_grad_norm=grad_clip,
            )
            dev_loss, dev_f1, dev_auc, dev_preds, dev_labels = evaluate_fusion(model, dev_loader, criterion, device)
        elif is_fusion_tcn:
            ga = (config or {}).get("training", {}).get("grad_accum_steps", 1)
            train_loss, train_grad_avg, train_grad_max = train_epoch_fusion_tcn(
                model, train_loader, optimizer, criterion, device,
                max_grad_norm=grad_clip, grad_accum_steps=ga,
            )
            dev_loss, dev_f1, dev_auc, dev_preds, dev_labels = evaluate_fusion_tcn(model, dev_loader, criterion, device)
        else:
            train_loss, train_grad_avg, train_grad_max = train_epoch(
                model, train_loader, optimizer, criterion, device, max_grad_norm=grad_clip,
            )
            dev_loss, dev_f1, dev_auc, dev_preds, dev_labels = evaluate(model, dev_loader, criterion, device)

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
            "dev/auc": dev_auc,
            "lr": current_lr,
        }
        metrics.update(_log_dev_metrics(epoch, dev_preds, dev_labels, class_names))
        metrics.update(_gpu_memory_mb())
        wandb.log(metrics)

        bar.set_postfix_str(f"loss={train_loss:.3f}/{dev_loss:.3f} f1={dev_f1:.3f} lr={current_lr:.1e}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir, "best_model.pth",
                model, optimizer, scheduler,
                epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
                dev_f1=dev_f1, best_dev_f1=best_f1, config=config,
            )
            wandb.summary["best_dev_f1"] = best_f1
            wandb.summary["best_epoch"] = epoch
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if save_frequency > 0 and epoch % save_frequency == 0:
            save_checkpoint(
                run_dir, f"checkpoint_epoch_{epoch:04d}.pth",
                model, optimizer, scheduler,
                epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
                dev_f1=dev_f1, best_dev_f1=best_f1, config=config,
            )

    wandb.finish()
    return {"best_dev_f1": best_f1, "history": history}
