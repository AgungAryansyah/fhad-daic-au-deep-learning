from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model_state", {})))
    return checkpoint.get("epoch", 0)


def find_latest_checkpoint(checkpoint_dir: Path, sweep_name: str | None = None, experiment_name: str | None = None) -> Path:
    candidates = []

    if sweep_name and experiment_name:
        p = checkpoint_dir / sweep_name / experiment_name / "latest"
        if p.is_symlink():
            candidates.append(p.resolve())

    if experiment_name:
        p = checkpoint_dir / experiment_name / "latest"
        if p.is_symlink():
            candidates.append(p.resolve())

    if sweep_name:
        p = checkpoint_dir / sweep_name / "latest"
        if p.is_symlink():
            candidates.append(p.resolve())

    p = checkpoint_dir / "latest"
    if p.is_symlink():
        candidates.append(p.resolve())

    for c in candidates:
        if c.is_dir():
            best = c / "best_model.pth"
            if best.exists():
                return best
        elif c.exists() and c.suffix == ".pth":
            return c

    return checkpoint_dir / "best.pt"


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.nan_to_num(torch.softmax(logits, dim=1).cpu(), nan=0.0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.tolist())
            all_probs.extend(probs.tolist())

    if class_names is None:
        class_names = [str(i) for i in range(max(all_labels) + 1)]

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    probs_arr = np.array(all_probs)
    row_sums = probs_arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs_arr = probs_arr / row_sums
    n_classes = probs_arr.shape[1]
    if n_classes == 2:
        auc = float(roc_auc_score(all_labels, probs_arr[:, 1]))
    else:
        auc = float(roc_auc_score(all_labels, probs_arr, multi_class="ovr", average="macro"))

    print("\n" + "=" * 50)
    print("Evaluation Report")
    print("=" * 50)
    print(report)
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("=" * 50 + "\n")

    return {"predictions": all_preds, "labels": all_labels, "confusion_matrix": cm, "auc": auc}
