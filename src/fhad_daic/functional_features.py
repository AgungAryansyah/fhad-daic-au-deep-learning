import numpy as np


def extract_functional_features(
    sessions: list[tuple[np.ndarray, int]]
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for X, y in sessions:
        if len(X) < 2:
            continue
        m = np.mean(X, axis=0)
        s = np.std(X, axis=0, ddof=1)
        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)
        t = np.arange(len(X), dtype=np.float32)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        slopes = np.zeros(X.shape[1], dtype=np.float32)
        if t_var > 0:
            slopes = ((t - t_mean) @ X) / t_var
        feats = np.concatenate([m, s, mn, mx, slopes]).astype(np.float32)
        features.append(feats)
        labels.append(y)
    return np.array(features), np.array(labels, dtype=np.int64)
