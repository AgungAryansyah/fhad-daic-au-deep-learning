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


def resolve_binary_label(df: pd.DataFrame, binning: dict | None) -> int:
    threshold = (binning or {}).get("binary_threshold")
    if threshold is not None:
        return int(float(df["phq_score"].iloc[0]) >= threshold)
    return int(df["phq_binary"].iloc[0])


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

    if modality == "egemaps":
        glob_pattern = "*_egemaps_clean.csv"
        fill_value = 0.0
    elif modality == "wav2vec":
        glob_pattern = "*_wav2vec_clean.csv"
        fill_value = 0.0
    else:
        glob_pattern = "*_clean.csv"
        fill_value = 0.5

    for csv_path in sorted(data_dir.glob(glob_pattern)):
        if modality == "au" and ("egemaps" in csv_path.name or "wav2vec" in csv_path.name):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        missing = [c for c in feature_cols if c not in df.columns]
        for c in missing:
            df[c] = fill_value
        X = df[feature_cols].values.astype(np.float32)
        if "confidence_mean" in feature_cols:
            ci = feature_cols.index("confidence")
            conf = pd.Series(X[:, ci])
            w = min(30, len(X))
            X[:, feature_cols.index("confidence_mean")] = conf.rolling(w, center=True, min_periods=1).mean().values
            s = conf.rolling(w, center=True, min_periods=1).std(ddof=0)
            X[:, feature_cols.index("confidence_std")] = s.fillna(0.0).values
            m = conf.rolling(w, center=True, min_periods=1).min()
            X[:, feature_cols.index("confidence_min")] = m.bfill().ffill().values
        if modality in ("egemaps", "wav2vec"):
            X = np.nan_to_num(X, nan=0.0)

        if mode == "multiclass":
            y = map_phq_to_bin(float(df["phq_score"].iloc[0]), bins)
        else:
            y = resolve_binary_label(df, binning)

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


def load_fusion_sessions(
    data_dir: Path,
    vis_feature_cols: list[str],
    aud_feature_cols: list[str],
    binning: dict | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    mode = resolve_label_mode(binning)
    bins = (binning or {}).get("bins", DEFAULT_BINS)

    vis_files = sorted([f for f in data_dir.glob("*_clean.csv") if "egemaps" not in f.name])
    aud_files = sorted(data_dir.glob("*_egemaps_clean.csv"))

    aud_by_sid = {}
    for ap in aud_files:
        sid = int(ap.stem.replace("_egemaps_clean", ""))
        aud_by_sid[sid] = ap

    vis_sessions = []
    aud_sessions = []
    labels = []

    aux_confidence = []
    aux_au_dyn = []
    aux_pose_var = []
    aux_hnr = []

    for vp in vis_files:
        sid = int(vp.stem.replace("_clean", ""))
        if sid not in aud_by_sid:
            continue
        ap = aud_by_sid[sid]

        vdf = pd.read_csv(vp)
        missing_v = [c for c in vis_feature_cols if c not in vdf.columns]
        for c in missing_v:
            vdf[c] = 0.5
        X_v = vdf[vis_feature_cols].values.astype(np.float32)
        cconf = vdf["confidence"].values.astype(np.float32) if "confidence" in vdf.columns else np.full(len(vdf), 0.5, dtype=np.float32)

        adf = pd.read_csv(ap)
        missing_a = [c for c in aud_feature_cols if c not in adf.columns]
        for c in missing_a:
            adf[c] = 0.0
        X_a = adf[aud_feature_cols].values.astype(np.float32)
        X_a = np.nan_to_num(X_a, nan=0.0)
        hnr_col = adf["HNRdBACF_sma3nz"].values.astype(np.float32) if "HNRdBACF_sma3nz" in adf.columns else np.zeros(len(adf), dtype=np.float32)
        hnr_col = np.nan_to_num(hnr_col, nan=0.0)

        if mode == "multiclass":
            y = map_phq_to_bin(float(vdf["phq_score"].iloc[0]), bins)
        else:
            y = resolve_binary_label(vdf, binning)

        vis_sessions.append(X_v)
        aud_sessions.append(X_a)
        labels.append(y)

        au_dyn = np.array([np.std(X_v[:, i]) for i in range(len(vis_feature_cols))])
        pose_cols_v = [i for i, c in enumerate(vis_feature_cols) if c.startswith("pose_")]
        pvar = np.mean([np.std(X_v[:, i]) for i in pose_cols_v]) if pose_cols_v else 0.0

        aux_confidence.append(float(np.mean(cconf)))
        aux_au_dyn.append(float(np.mean(au_dyn)))
        aux_pose_var.append(float(pvar))
        aux_hnr.append(float(np.mean(hnr_col)))

    aux_v = {
        "confidence_mean": np.array(aux_confidence, dtype=np.float32),
        "au_dyn_mean": np.array(aux_au_dyn, dtype=np.float32),
        "pose_var_mean": np.array(aux_pose_var, dtype=np.float32),
    }
    aux_a = {
        "hnr_mean": np.array(aux_hnr, dtype=np.float32),
    }

    return vis_sessions, aud_sessions, np.array(labels, dtype=np.int64), aux_v, aux_a


def load_fusion_tcn_sessions(
    data_dir: Path,
    vis_feature_cols: list[str],
    aud_feature_cols: list[str],
    binning: dict | None = None,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    mode = resolve_label_mode(binning)
    bins = (binning or {}).get("bins", DEFAULT_BINS)

    vis_files = sorted([f for f in data_dir.glob("*_clean.csv") if "egemaps" not in f.name])
    aud_files = sorted(data_dir.glob("*_egemaps_clean.csv"))

    aud_by_sid = {}
    for ap in aud_files:
        sid = int(ap.stem.replace("_egemaps_clean", ""))
        aud_by_sid[sid] = ap

    vis_sessions = []
    aud_sessions = []
    labels = []

    aux_confidence = []
    aux_au_dyn = []
    aux_pose_var = []
    aux_hnr = []

    has_conf_agg = "confidence_mean" in vis_feature_cols
    ci = vis_feature_cols.index("confidence") if "confidence" in vis_feature_cols else None

    for vp in vis_files:
        sid = int(vp.stem.replace("_clean", ""))
        if sid not in aud_by_sid:
            continue
        ap = aud_by_sid[sid]

        vdf = pd.read_csv(vp)
        missing_v = [c for c in vis_feature_cols if c not in vdf.columns]
        for c in missing_v:
            vdf[c] = 0.5
        X_v = vdf[vis_feature_cols].values.astype(np.float32)
        if max_frames and len(X_v) > max_frames:
            X_v = X_v[-max_frames:]
        cconf = vdf["confidence"].values.astype(np.float32) if "confidence" in vdf.columns else np.full(len(vdf), 0.5, dtype=np.float32)
        if max_frames and len(cconf) > max_frames:
            cconf = cconf[-max_frames:]

        if has_conf_agg and ci is not None:
            conf = pd.Series(X_v[:, ci])
            w = min(30, len(X_v))
            X_v[:, vis_feature_cols.index("confidence_mean")] = conf.rolling(w, center=True, min_periods=1).mean().values
            s = conf.rolling(w, center=True, min_periods=1).std(ddof=0)
            X_v[:, vis_feature_cols.index("confidence_std")] = s.fillna(0.0).values
            m = conf.rolling(w, center=True, min_periods=1).min()
            X_v[:, vis_feature_cols.index("confidence_min")] = m.bfill().ffill().values

        adf = pd.read_csv(ap)
        missing_a = [c for c in aud_feature_cols if c not in adf.columns]
        for c in missing_a:
            adf[c] = 0.0
        X_a = adf[aud_feature_cols].values.astype(np.float32)
        if max_frames and len(X_a) > max_frames:
            X_a = X_a[-max_frames:]
        X_a = np.nan_to_num(X_a, nan=0.0)
        hnr_col = adf["HNRdBACF_sma3nz"].values.astype(np.float32) if "HNRdBACF_sma3nz" in adf.columns else np.zeros(len(adf), dtype=np.float32)
        if max_frames and len(hnr_col) > max_frames:
            hnr_col = hnr_col[-max_frames:]
        hnr_col = np.nan_to_num(hnr_col, nan=0.0)

        if mode == "multiclass":
            y = map_phq_to_bin(float(vdf["phq_score"].iloc[0]), bins)
        else:
            y = resolve_binary_label(vdf, binning)

        vis_sessions.append(X_v)
        aud_sessions.append(X_a)
        labels.append(y)

        au_dyn = np.array([np.std(X_v[:, i]) for i in range(len(vis_feature_cols))])
        pose_cols_v = [i for i, c in enumerate(vis_feature_cols) if c.startswith("pose_")]
        pvar = np.mean([np.std(X_v[:, i]) for i in pose_cols_v]) if pose_cols_v else 0.0

        aux_confidence.append(float(np.mean(cconf)))
        aux_au_dyn.append(float(np.mean(au_dyn)))
        aux_pose_var.append(float(pvar))
        aux_hnr.append(float(np.mean(hnr_col)))

    aux_v = {
        "confidence_mean": np.array(aux_confidence, dtype=np.float32),
        "au_dyn_mean": np.array(aux_au_dyn, dtype=np.float32),
        "pose_var_mean": np.array(aux_pose_var, dtype=np.float32),
    }
    aux_a = {
        "hnr_mean": np.array(aux_hnr, dtype=np.float32),
    }

    return vis_sessions, aud_sessions, np.array(labels, dtype=np.int64), aux_v, aux_a


class FusionTCNDataset(Dataset):
    def __init__(
        self,
        vis_sessions: list[np.ndarray],
        aud_sessions: list[np.ndarray],
        labels: np.ndarray,
        aux_v: dict[str, np.ndarray],
        aux_a: dict[str, np.ndarray],
    ):
        self.vis_sessions = vis_sessions
        self.aud_sessions = aud_sessions
        self.labels = torch.from_numpy(labels)
        self.aux_v = {k: torch.from_numpy(v) for k, v in aux_v.items()}
        self.aux_a = {k: torch.from_numpy(v) for k, v in aux_a.items()}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.vis_sessions[idx]),
            torch.from_numpy(self.aud_sessions[idx]),
            self.labels[idx],
            {k: v[idx] for k, v in self.aux_v.items()},
            {k: v[idx] for k, v in self.aux_a.items()},
        )


def collate_fusion_tcn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    X_v_list = [item[0] for item in batch]
    X_a_list = [item[1] for item in batch]
    y_list = [item[2] for item in batch]
    aux_v_list = [item[3] for item in batch]
    aux_a_list = [item[4] for item in batch]

    y = torch.stack(y_list)

    X_v = torch.nn.utils.rnn.pad_sequence(X_v_list, batch_first=True)
    mask_v = torch.zeros(len(batch), X_v.size(1), dtype=torch.bool)
    for i, x in enumerate(X_v_list):
        mask_v[i, :len(x)] = True

    X_a = torch.nn.utils.rnn.pad_sequence(X_a_list, batch_first=True)
    mask_a = torch.zeros(len(batch), X_a.size(1), dtype=torch.bool)
    for i, x in enumerate(X_a_list):
        mask_a[i, :len(x)] = True

    aux_v = {k: torch.stack([a[k] for a in aux_v_list]) for k in aux_v_list[0]}
    aux_a = {k: torch.stack([a[k] for a in aux_a_list]) for k in aux_a_list[0]}

    return X_v, mask_v, X_a, mask_a, y, aux_v, aux_a


class FusionDataset(Dataset):
    def __init__(
        self,
        F_v: np.ndarray,
        F_a: np.ndarray,
        y: np.ndarray,
        aux_v: dict[str, np.ndarray],
        aux_a: dict[str, np.ndarray],
    ):
        self.F_v = torch.from_numpy(F_v)
        self.F_a = torch.from_numpy(F_a)
        self.y = torch.from_numpy(y)
        self.aux_v = {k: torch.from_numpy(v) for k, v in aux_v.items()}
        self.aux_a = {k: torch.from_numpy(v) for k, v in aux_a.items()}

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            self.F_v[idx],
            self.F_a[idx],
            self.y[idx],
            {k: v[idx] for k, v in self.aux_v.items()},
            {k: v[idx] for k, v in self.aux_a.items()},
        )
