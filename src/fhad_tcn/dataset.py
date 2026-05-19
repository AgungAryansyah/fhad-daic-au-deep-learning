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
    sessions: list[tuple[np.ndarray, int]], window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    windows, labels = [], []
    for X, y in sessions:
        for start in range(0, len(X) - window_size + 1, stride):
            windows.append(X[start : start + window_size])
            labels.append(y)
    return np.stack(windows), np.array(labels, dtype=np.int64)


class AUWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.from_numpy(windows)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, a_min=1, a_max=None)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights)
