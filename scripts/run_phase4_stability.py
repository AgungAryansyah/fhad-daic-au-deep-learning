import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_daic.config import get_feature_cols
from fhad_daic.data import (
    AUWindowDataset,
    apply_scaler,
    fit_scaler,
    load_sessions,
    resolve_label_mode,
    resolve_modality,
    slide_windows,
)
from fhad_daic.functional_features import extract_functional_features
from fhad_daic.metrics import PERTURBATION_TYPES, compute_pgi_pgu_empirical
from fhad_daic.models import GRUModel, LSTMModel, MLP, MILTCN, TCN

DEFAULT_KS = [5]
N_BG = 50


def list_checkpoints(sweep_dir: Path) -> list[tuple[str, Path]]:
    results = []
    for exp_dir in sorted(sweep_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        latest = exp_dir / "latest"
        if latest.is_symlink():
            cp = latest.resolve() / "best_model.pth"
            if cp.exists():
                results.append((exp_dir.name, cp))
    return results


def build_model(cfg: dict, num_inputs: int, device: torch.device):
    t_cfg = cfg["training"]
    mt = t_cfg.get("model_type", "tcn")
    nc = t_cfg["num_classes"]
    if mt == "gru":
        rnn_cfg = cfg["gru"]
        return GRUModel(num_inputs, rnn_cfg["hidden_size"], rnn_cfg["num_layers"],
                        rnn_cfg["dropout"], nc, rnn_cfg.get("bidirectional", True))
    elif mt == "lstm":
        rnn_cfg = cfg["lstm"]
        return LSTMModel(num_inputs, rnn_cfg["hidden_size"], rnn_cfg["num_layers"],
                          rnn_cfg["dropout"], nc, rnn_cfg.get("bidirectional", True))
    elif mt == "functional":
        mlp_cfg = cfg.get("mlp", {})
        return MLP(num_inputs * 5, mlp_cfg.get("hidden_dims", [64, 32]),
                    mlp_cfg.get("dropout", 0.7), nc)
    elif mt == "mil":
        tcn_cfg = cfg["tcn"]
        attn_dim = cfg.get("mil", {}).get("attn_dim", 64)
        return MILTCN(num_inputs, tcn_cfg["num_channels"], tcn_cfg["kernel_size"],
                       tcn_cfg["dropout"], nc, attn_dim)
    else:
        tcn_cfg = cfg["tcn"]
        return TCN(num_inputs, tcn_cfg["num_channels"], tcn_cfg["kernel_size"],
                    tcn_cfg["dropout"], nc)


def run_shap_importance(model, X: np.ndarray, feature_cols: list[str], device: torch.device,
                        model_type: str, n_test: int = 10, n_bg: int = 50) -> dict[str, float]:
    if model_type == "mil":
        return {}

    bg_idx = np.random.choice(len(X), size=min(n_bg, len(X)), replace=False)
    bg_t = torch.from_numpy(X[bg_idx]).float().to(device)
    test_idx = np.random.choice(len(X), size=min(n_test, len(X)), replace=False)
    test_t = torch.from_numpy(X[test_idx]).float().to(device)

    explainer = shap.GradientExplainer(model, bg_t)
    shap_values = explainer.shap_values(test_t)

    sv = np.asarray(shap_values)

    # Multi-output explainers return a list of arrays (one per class).
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])

    # If the last dimension is the number of classes, extract class 1.
    if sv.ndim >= 3 and sv.shape[-1] > 1 and sv.shape[-1] != len(feature_cols):
        sv = sv[..., 1]

    sv = np.asarray(sv)

    # Aggregate to one scalar per feature. Temporal models produce
    # (n_samples, n_features, window_size) SHAP values.
    if sv.ndim == 3:
        if sv.shape[1] == len(feature_cols):
            feat_imp = np.abs(sv).mean(axis=(0, 2))
        elif sv.shape[2] == len(feature_cols):
            feat_imp = np.abs(sv).mean(axis=(0, 1))
        else:
            feat_imp = np.abs(sv).mean(axis=(0, 1))
    elif sv.ndim == 2:
        feat_imp = np.abs(sv).mean(axis=0)
    elif sv.ndim == 1:
        feat_imp = np.abs(sv)
    else:
        feat_imp = np.array([])

    def _to_scalar(v):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            v = v.squeeze()
            if v.numel() > 1:
                return float(v.mean())
            return float(v.item())
        return float(v)

    # Functional models have 5 stats per raw feature (mean, std, min, max, slope).
    # Collapse them to one importance score per original feature.
    if model_type == "functional" and len(feat_imp) == len(feature_cols) * 5:
        feat_imp = np.array([
            float(feat_imp[i * 5:(i + 1) * 5].mean())
            for i in range(len(feature_cols))
        ])

    return {
        feature_cols[i]: _to_scalar(feat_imp[i])
        for i in range(len(feature_cols))
        if i < len(feat_imp)
    }


def _slide_windows_limited(sessions, window_size, stride, max_windows):
    """Slide windows but stop once max_windows is reached to avoid huge arrays."""
    windows, labels = [], []
    for X, y in sessions:
        n = (len(X) - window_size) // stride + 1
        if n <= 0:
            continue
        for start in range(0, len(X) - window_size + 1, stride):
            windows.append(X[start : start + window_size])
            labels.append(y)
            if len(windows) >= max_windows:
                return np.stack(windows), np.array(labels, dtype=np.int64)
    if not windows:
        return np.array([]), np.array([])
    return np.stack(windows), np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Batch stability/PGI/PGU analysis on sweep checkpoints")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name in checkpoints/ dir")
    parser.add_argument("--output", type=str, default=None, help="Output CSV (default: results/stability_{sweep}.csv)")
    parser.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS, help="Top-k values for feature masking")
    parser.add_argument("--n-test", type=int, default=20, help="Number of test windows for SHAP")
    parser.add_argument("--n-bg", type=int, default=N_BG, help="Number of background samples for SHAP")
    parser.add_argument("--max-train-sessions", type=int, default=30, help="Max train sessions to slide")
    parser.add_argument("--max-dev-sessions", type=int, default=30, help="Max dev sessions to slide")
    parser.add_argument("--max-train-windows", type=int, default=500, help="Max train windows/sessions for SHAP")
    parser.add_argument("--max-dev-windows", type=int, default=1000, help="Max dev windows/sessions for PGI/PGU")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device_str = args.device
    if device_str in ("gpu", "cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sweep_dir = Path("checkpoints") / args.sweep
    if not sweep_dir.exists():
        print(f"Sweep not found: {sweep_dir}")
        return

    checkpoints = list_checkpoints(sweep_dir)
    if not checkpoints:
        print("No checkpoints found")
        return
    print(f"Found {len(checkpoints)} experiment(s)")

    output_path = Path(args.output) if args.output else Path("results") / f"stability_{args.sweep}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["experiment", "modality", "model_type", "baseline_f1", "top_k", "pgi", "pgu", "pgi_pgu_ratio", "stability"]
    for ptype in PERTURBATION_TYPES:
        fieldnames.extend([f"f1_drop_{ptype}", f"stability_{ptype}"])
    fieldnames.append("f1_drop_mask_random")
    rows = []

    for exp_name, cp_path in tqdm(checkpoints, desc="Stability analysis"):
        raw = torch.load(cp_path, map_location="cpu", weights_only=False)
        cfg = raw.get("config")
        if not cfg:
            continue

        t_cfg = cfg.get("training", {})
        model_type = t_cfg.get("model_type", "tcn")
        if model_type in ("fusion", "fusion_tcn", "concat_fusion", "concat_fusion_tcn", "mil"):
            continue

        binning = cfg.get("binning")
        modality = resolve_modality(cfg.get("features"))
        feature_cols = get_feature_cols(cfg)

        try:
            sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols, binning=binning, modality=modality)
            dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols, binning=binning, modality=modality)

            # Keep memory bounded: only use a subset of sessions.
            if len(dev_sessions) > args.max_dev_sessions:
                idx = np.random.choice(len(dev_sessions), args.max_dev_sessions, replace=False)
                dev_sessions = [dev_sessions[i] for i in idx]
            if len(sessions) > args.max_train_sessions:
                idx = np.random.choice(len(sessions), args.max_train_sessions, replace=False)
                sessions = [sessions[i] for i in idx]

            scaler = fit_scaler(sessions)
            dev_sessions = apply_scaler(dev_sessions, scaler)

            if model_type == "functional":
                train_sessions = apply_scaler(sessions, scaler)
                X_train, _ = extract_functional_features(train_sessions)
                X_dev, y_dev = extract_functional_features(dev_sessions)

                if len(X_train) > args.max_train_windows:
                    idx = np.random.choice(len(X_train), args.max_train_windows, replace=False)
                    X_train = X_train[idx]
                if len(X_dev) > args.max_dev_windows:
                    idx = np.random.choice(len(X_dev), args.max_dev_windows, replace=False)
                    X_dev = X_dev[idx]
                    y_dev = y_dev[idx]
            else:
                w_cfg = cfg["windowing"]
                X_dev, y_dev = _slide_windows_limited(
                    dev_sessions, w_cfg["window_size"], w_cfg["stride"], args.max_dev_windows
                )
                train_sessions = apply_scaler(sessions, scaler)
                X_train, _ = _slide_windows_limited(
                    train_sessions, w_cfg["window_size"], w_cfg["stride"], args.max_train_windows
                )

            model = build_model(cfg, len(feature_cols), device).to(device)
            msd = raw.get("model_state_dict", raw.get("model_state", {}))
            model.load_state_dict(msd)
            model.eval()

            # Get SHAP importance once (use train data)
            importance = run_shap_importance(model, X_train, feature_cols, device, model_type,
                                             n_test=args.n_test, n_bg=args.n_bg)
            if not importance:
                continue
        except Exception as e:
            print(f"\n  SKIP {exp_name}: {e}")
            continue

        # Convert dev windows to list of arrays for perturbation
        X_dev_list = [X_dev]

        def _make_loader(X_list):
            X_stacked = np.concatenate(X_list)
            y_stacked = np.tile(y_dev, len(X_list))
            return DataLoader(AUWindowDataset(X_stacked, y_stacked), batch_size=min(256, len(X_stacked)))

        for top_k in args.ks:
            n_classes = t_cfg["num_classes"]
            counts = np.bincount(y_dev, minlength=n_classes).astype(np.float32)
            counts = np.clip(counts, a_min=1, a_max=None)
            cw = torch.from_numpy(counts.sum() / (n_classes * counts)).to(device)
            criterion = nn.CrossEntropyLoss(weight=cw)

            results = compute_pgi_pgu_empirical(
                model, _make_loader, X_dev_list, feature_cols,
                importance, top_k, criterion, device,
            )

            row = {
                "experiment": exp_name,
                "modality": modality,
                "model_type": model_type,
                "baseline_f1": results["baseline_f1"],
                "top_k": top_k,
                "pgi": results["pgi"],
                "pgu": results["pgu"],
                "pgi_pgu_ratio": results["pgi_pgu_ratio"],
                "stability": results["stability"],
            }
            for ptype in PERTURBATION_TYPES:
                row[f"f1_drop_{ptype}"] = results.get(f"f1_drop_{ptype}", 0)
                row[f"stability_{ptype}"] = results.get(f"stability_{ptype}", 0)
            row["f1_drop_mask_random"] = results.get("f1_drop_mask_random", 0)
            rows.append(row)

        del model, sessions, dev_sessions, train_sessions, X_train, X_dev, y_dev, scaler, importance, X_dev_list
        gc.collect()
        torch.cuda.empty_cache()

    if not rows:
        print("No results.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {output_path}")

    print(f"\n{'experiment':<30s} {'F1':>6s} {'PGI':>6s} {'PGU':>6s} {'ratio':>6s} {'stab':>6s}")
    for r in sorted(rows, key=lambda r: r.get("pgi_pgu_ratio", 0), reverse=True):
        k = r.get("top_k", "?")
        print(f"{r['experiment']:<30s} {r['baseline_f1']:>6.4f} {r['pgi']:>6.4f} {r['pgu']:>6.4f} {r['pgi_pgu_ratio']:>6.2f} {r['stability']:>6.4f}")


if __name__ == "__main__":
    main()
