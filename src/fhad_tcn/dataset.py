from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def load_sessions(data_dir: Path, feature_cols: list[str]) -> list[tuple[np.ndarray, int]]:
    sessions = []
    for csv_path in sorted(data_dir.glob("*_clean.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            continue
        X = df[feature_cols].values.astype(np.float32)
        y = int(df["phq_binary"].iloc[0])
        sessions.append((X, y))
    return sessions


def fit_scaler(sessions: list[tuple[np.ndarray, int]]) -> StandardScaler:
    all_frames = np.concatenate([X for X, _ in sessions], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_frames)
    return scaler


def apply_scaler(
    sessions: list[tuple[np.ndarray, int]], scaler: StandardScaler
) -> list[tuple[np.ndarray, int]]:
    return [(scaler.transform(X), y) for X, y in sessions]


def slide_windows(
    sessions: list[tuple[np.ndarray, int]], window_size: int, stride: int, return_sids: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, labels = [], []
    sids = [] if return_sids else None
    for sid, (X, y) in enumerate(sessions):
        n_windows = (len(X) - window_size) // stride + 1
        if n_windows <= 0:
            continue
        for start in range(0, len(X) - window_size + 1, stride):
            windows.append(X[start : start + window_size])
            labels.append(y)
        if return_sids:
            sids.extend([sid] * n_windows)
    X_out = np.stack(windows)
    y_out = np.array(labels, dtype=np.int64)
    if return_sids:
        return X_out, y_out, np.array(sids, dtype=np.int64)
    return X_out, y_out


class AUWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.from_numpy(windows)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


class MILWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, session_ids: np.ndarray):
        self.windows = torch.from_numpy(windows)
        self.labels = torch.from_numpy(labels)
        self.session_ids = torch.from_numpy(session_ids)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx], self.session_ids[idx]


def collate_mil(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    sids = torch.stack([b[2] for b in batch])
    return X, y, sids


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, a_min=1, a_max=None)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights)
