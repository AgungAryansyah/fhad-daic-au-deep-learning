from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .data import map_phq_to_bin, resolve_label_mode
from .functional_features import extract_functional_features
from .models.mlp import MLP
from .training import get_class_names


def load_sessions_with_ids(
    data_dirs: list[Path], feature_cols: list[str], binning: dict | None = None
) -> dict[int, tuple[np.ndarray, int]]:
    mode = resolve_label_mode(binning)
    bins = (binning or {}).get("bins", [])

    sessions = {}
    for data_dir in data_dirs:
        for csv_path in sorted(data_dir.glob("*_clean.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                continue
            X = df[feature_cols].values.astype(np.float32)

            if mode == "multiclass" and bins:
                y = map_phq_to_bin(float(df["phq_score"].iloc[0]), bins)
            else:
                y = int(df["phq_binary"].iloc[0])

            sid = int(csv_path.stem.replace("_clean", ""))
            sessions[sid] = (X, y)
    return sessions


def _fit_scaler(sessions: list[tuple[np.ndarray, int]]) -> StandardScaler:
    all_frames = np.concatenate([X for X, _ in sessions], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_frames)
    return scaler


def _apply_scaler(
    sessions: list[tuple[np.ndarray, int]], scaler: StandardScaler
) -> list[tuple[np.ndarray, int]]:
    return [(scaler.transform(X), y) for X, y in sessions]


def _train_mlp_on_fold(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
) -> float:
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return criterion(model(X), y).item()


def run_loso_cv_mlp(
    train_dir: Path,
    dev_dir: Path,
    feature_cols: list[str],
    cfg: dict,
    device: torch.device,
) -> dict:
    binning = cfg.get("binning")
    all_sessions = load_sessions_with_ids([train_dir, dev_dir], feature_cols, binning=binning)
    all_sids = sorted(all_sessions.keys())
    n = len(all_sids)
    print(f"\nLOSO CV: {n} labeled sessions")

    class_names = get_class_names(binning)

    mlp_cfg = cfg.get("mlp", {})
    t_cfg = cfg["training"]
    num_epochs = t_cfg.get("num_epochs", 200)
    lr = t_cfg.get("learning_rate", 0.001)
    weight_decay = mlp_cfg.get("weight_decay", 0.01)
    dropout = mlp_cfg.get("dropout", 0.7)
    hidden_dims = mlp_cfg.get("hidden_dims", [64, 32])
    num_features = len(feature_cols) * 5

    fold_results = []
    all_preds = []
    all_labels = []

    for test_sid in tqdm(all_sids, desc="LOSO CV"):
        train_sessions = [all_sessions[sid] for sid in all_sids if sid != test_sid]
        test_session = all_sessions[test_sid]

        scaler = _fit_scaler(train_sessions)
        train_scaled = _apply_scaler(train_sessions, scaler)
        test_scaled = _apply_scaler([test_session], scaler)

        train_X_raw, train_y_raw = extract_functional_features(train_scaled)
        test_X_raw, test_y_raw = extract_functional_features(test_scaled)

        train_X = torch.from_numpy(train_X_raw).to(device)
        train_y = torch.from_numpy(train_y_raw).to(device)
        test_X = torch.from_numpy(test_X_raw).to(device)
        test_y = torch.from_numpy(test_y_raw).to(device)

        model = MLP(
            num_features=num_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=t_cfg["num_classes"],
        ).to(device)

        class_counts = np.bincount(train_y_raw, minlength=t_cfg["num_classes"])
        class_counts = np.clip(class_counts.astype(np.float32), a_min=1, a_max=None)
        class_weights = torch.from_numpy(class_counts.sum() / (t_cfg["num_classes"] * class_counts)).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=t_cfg.get("label_smoothing", 0.1))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        _train_mlp_on_fold(model, train_X, train_y, optimizer, criterion, num_epochs, device)

        with torch.no_grad():
            logits = model(test_X)
            pred = int(logits.argmax(dim=1).item())
            label = int(test_y.item())

        fold_results.append({"sid": test_sid, "pred": pred, "label": label})
        all_preds.append(pred)
        all_labels.append(label)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"LOSO CV Results ({n} folds)")
    print(f"{'='*50}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion matrix:\n{cm}")

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "n_folds": n,
        "folds": fold_results,
    }
