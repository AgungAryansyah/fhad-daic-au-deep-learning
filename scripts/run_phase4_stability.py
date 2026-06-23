import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import shap
import torch
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
        return MLP(num_inputs, mlp_cfg.get("hidden_dims", [64, 32]),
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
                        model_type: str, n_test: int = 10) -> dict[str, float]:
    bg_idx = np.random.choice(len(X), size=min(N_BG, len(X)), replace=False)
    bg_t = torch.from_numpy(X[bg_idx]).float().to(device)
    test_idx = np.random.choice(len(X), size=min(n_test, len(X)), replace=False)
    test_t = torch.from_numpy(X[test_idx]).float().to(device)

    if model_type == "functional":
        explainer = shap.KernelExplainer(
            lambda x: model(torch.from_numpy(x).to(device)).detach().cpu().numpy(),
            X[bg_idx].astype(np.float32),
            link="identity",
        )
        shap_values = explainer.shap_values(test_t.cpu().numpy()[:min(n_test, 10)],
                                             nsamples=min(2 * len(feature_cols), 100))
    elif model_type == "mil":
        return {}
    else:
        explainer = shap.GradientExplainer(model, bg_t)
        shap_values = explainer.shap_values(test_t)

    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif shap_values.ndim == 3:
        sv_arr = np.array(shap_values)
        sv = sv_arr[:, :, 1] if sv_arr.shape[2] > 1 else sv_arr
    else:
        sv = np.array(shap_values)

    if sv.ndim == 3:
        sv = sv.mean(axis=1)
    feat_imp = np.abs(sv).mean(axis=0)
    return {feature_cols[i]: float(feat_imp[i]) for i in range(len(feature_cols))}


def main():
    parser = argparse.ArgumentParser(description="Batch stability/PGI/PGU analysis on sweep checkpoints")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name in checkpoints/ dir")
    parser.add_argument("--output", type=str, default=None, help="Output CSV (default: results/stability_{sweep}.csv)")
    parser.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS, help="Top-k values for feature masking")
    parser.add_argument("--n-test", type=int, default=20, help="Number of test windows for SHAP")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
        if model_type in ("fusion", "fusion_tcn", "concat_fusion", "concat_fusion_tcn"):
            continue

        binning = cfg.get("binning")
        modality = resolve_modality(cfg.get("features"))
        feature_cols = get_feature_cols(cfg)

        try:
            sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols, binning=binning, modality=modality)
            dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols, binning=binning, modality=modality)
            scaler = fit_scaler(sessions)
            dev_sessions = apply_scaler(dev_sessions, scaler)
            w_cfg = cfg["windowing"]
            X_dev, y_dev = slide_windows(dev_sessions, w_cfg["window_size"], w_cfg["stride"])

            model = build_model(cfg, len(feature_cols), device).to(device)
            msd = raw.get("model_state_dict", raw.get("model_state", {}))
            model.load_state_dict(msd)
            model.eval()

            # Get SHAP importance once (use train data)
            train_sessions = apply_scaler(sessions, scaler)
            X_train, _ = slide_windows(train_sessions, w_cfg["window_size"], w_cfg["stride"])
            importance = run_shap_importance(model, X_train, feature_cols, device, model_type, n_test=args.n_test)
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
            import torch.nn as nn
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
