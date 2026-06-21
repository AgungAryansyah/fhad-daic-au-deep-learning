from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model_state", {})))
    return checkpoint.get("epoch", 0)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    latest_dir = checkpoint_dir / "latest"
    if latest_dir.is_symlink():
        best = latest_dir / "best_model.pth"
        if best.exists():
            return best.resolve()
    return checkpoint_dir / "best.pt"


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    if class_names is None:
        class_names = [str(i) for i in range(max(all_labels) + 1)]

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 50)
    print("Evaluation Report")
    print("=" * 50)
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 50 + "\n")

    return {"predictions": all_preds, "labels": all_labels, "confusion_matrix": cm}
