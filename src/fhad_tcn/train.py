import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


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
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
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
    return avg_loss, macro_f1


def train_epoch_mil(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_bags = 0
    for X, y, sids in loader:
        X, y, sids = X.to(device), y.to(device), sids.to(device)
        optimizer.zero_grad()
        logits = model(X, sids)
        bag_y = get_bag_labels(y, sids)
        loss = criterion(logits, bag_y)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(bag_y)
        total_bags += len(bag_y)
    return total_loss / max(total_bags, 1)


def evaluate_mil(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
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
    return avg_loss, macro_f1


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

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "fhad-tcn-depression"),
        entity=os.getenv("WANDB_ENTITY"),
        mode=os.getenv("WANDB_MODE", "online"),
        config=config,
        name="tcn-baseline",
        tags=["tcn", "depression", "au-features"],
    )

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

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        if is_mil:
            train_loss = train_epoch_mil(model, train_loader, optimizer, criterion, device, max_grad_norm=grad_clip)
            dev_loss, dev_f1 = evaluate_mil(model, dev_loader, criterion, device)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=grad_clip)
            dev_loss, dev_f1 = evaluate(model, dev_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(dev_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_f1": dev_f1, "lr": current_lr})
        tqdm.write(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | dev_f1={dev_f1:.4f} | lr={current_lr:.2e}")

        wandb.log({"epoch": epoch, "train/loss": train_loss, "dev/loss": dev_loss, "dev/macro_f1": dev_f1, "lr": current_lr})

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
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break

    wandb.finish()
    return {"best_dev_f1": best_f1, "history": history}
