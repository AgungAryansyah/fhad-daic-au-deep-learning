from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    run_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    dev_f1: float,
    best_dev_f1: float,
    config: dict | None = None,
) -> Path:
    path = run_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_f1": dev_f1,
            "best_dev_f1": best_dev_f1,
            "config": config,
        },
        path,
    )
    return path


def load_full_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: torch.device | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
