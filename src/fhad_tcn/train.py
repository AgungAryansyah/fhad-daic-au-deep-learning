from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
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
) -> dict:
    best_f1 = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_f1 = evaluate(model, dev_loader, criterion, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_f1": dev_f1})
        tqdm.write(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | dev_f1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "dev_f1": dev_f1}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break

    return {"best_dev_f1": best_f1, "history": history}
