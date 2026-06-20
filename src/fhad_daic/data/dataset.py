import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

CACHE_DIR = Path(os.getenv("WINDOW_CACHE_DIR", "cache/windowed"))

DEFAULT_BINS = [
    {"label": 0, "name": "minimal",   "min": 0,  "max": 4},
    {"label": 1, "name": "mild",      "min": 5,  "max": 9},
    {"label": 2, "name": "moderate",  "min": 10, "max": 14},
    {"label": 3, "name": "severe",    "min": 15, "max": 24},
]


def map_phq_to_bin(phq_score: float, bins: list[dict]) -> int:
    for b in bins:
        if b["min"] <= phq_score <= b["max"]:
            return b["label"]
    return len(bins) - 1


def resolve_label_mode(binning: dict | None) -> str:
    if binning and binning.get("mode") == "phq_multiclass":
        return "multiclass"
    return "binary"


def resolve_modality(features_cfg: dict | None) -> str:
    return (features_cfg or {}).get("modality", "au")


def get_window_cache_path(split: str, window_size: int, stride: int, mil: bool = False, label_mode: str = "binary", modality: str = "au") -> Path:
    prefix = f"{split}_mil" if mil else split
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{window_size}_{stride}_{label_mode}_{modality}.pkl"


def load_sessions(data_dir: Path, feature_cols: list[str], binning: dict | None = None, modality: str = "au") -> list[tuple[np.ndarray, int]]:
    sessions = []
    mode = resolve_label_mode(binning)
    bins = (binning or {}).get("bins", DEFAULT_BINS)

    glob_pattern = "*_egemaps_clean.csv" if modality == "egemaps" else "*_clean.csv"

    for csv_path in sorted(data_dir.glob(glob_pattern)):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            continue
        X = df[feature_cols].values.astype(np.float32)
        if modality == "egemaps":
            X = np.nan_to_num(X, nan=0.0)

        if mode == "multiclass":
            y = map_phq_to_bin(float(df["phq_score"].iloc[0]), bins)
        else:
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
