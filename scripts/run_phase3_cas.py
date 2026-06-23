import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import shap
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_daic.config import get_feature_cols
from fhad_daic.data import (
    apply_scaler,
    fit_scaler,
    load_sessions,
    resolve_label_mode,
    resolve_modality,
    slide_windows,
)
from fhad_daic.metrics import CLINICAL_AUDIO_FEATURES, CLINICAL_VISUAL_FEATURES, compute_cas_at_k
from fhad_daic.models import GRUModel, LSTMModel, MLP, MILTCN, TCN

DEFAULT_KS = [1, 3, 5, 10, 20]
N_BG = 50
N_TEST = 10


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


def load_data_for_shap(cfg: dict, feature_cols: list[str], modality: str, binning, n_max: int = 200):
    train_dir = Path(cfg["data"]["train_dir"])

    sessions = load_sessions(train_dir, feature_cols, binning=binning, modality=modality)
    if not sessions:
        raise RuntimeError("No sessions loaded")

    if cfg.get("training", {}).get("model_type") == "functional":
        from fhad_daic.functional_features import extract_functional_features
        features, labels = extract_functional_features(sessions)
        return features[:n_max], labels[:n_max], feature_cols

    scaler = fit_scaler(sessions)
    sessions = apply_scaler(sessions, scaler)
    w_cfg = cfg["windowing"]
    X, y = slide_windows(sessions, w_cfg["window_size"], w_cfg["stride"])
    return X[:n_max], y[:n_max], feature_cols


def run_shap_on_model(
    model, X: np.ndarray, feature_cols: list[str], device: torch.device, model_type: str
) -> dict[str, float]:
    if model_type == "functional":
        bg_idx = np.random.choice(len(X), size=min(N_BG, len(X)), replace=False)
        bg = X[bg_idx].astype(np.float32)
        test_idx = np.random.choice(len(X), size=min(N_TEST, len(X)), replace=False)
        test = X[test_idx].astype(np.float32)
        explainer = shap.KernelExplainer(lambda x: model(torch.from_numpy(x).to(device)).detach().cpu().numpy(),
                                          bg, link="identity")
        shap_values = explainer.shap_values(test[:min(N_TEST, 10)], nsamples=min(2 * len(feature_cols), 100))
    elif model_type == "mil":
        return {}
    else:
        bg_idx = np.random.choice(len(X), size=min(N_BG, len(X)), replace=False)
        bg_t = torch.from_numpy(X[bg_idx]).float().to(device)
        test_idx = np.random.choice(len(X), size=min(N_TEST, len(X)), replace=False)
        test_t = torch.from_numpy(X[test_idx]).float().to(device)
        explainer = shap.GradientExplainer(model, bg_t)
        shap_values = explainer.shap_values(test_t)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv_arr = np.array(shap_values)
        if sv_arr.ndim == 3 and sv_arr.shape[2] > 1:
            sv = sv_arr[:, :, 1]
        else:
            sv = sv_arr
    else:
        sv = np.array(shap_values)

    if sv.ndim == 3:
        sv = sv.mean(axis=1)
    feat_imp = np.abs(sv).mean(axis=0)
    return {feature_cols[i]: float(feat_imp[i]) for i in range(len(feature_cols))}


def main():
    parser = argparse.ArgumentParser(description="Batch CAS analysis on sweep checkpoints")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name in checkpoints/ dir")
    parser.add_argument("--output", type=str, default=None, help="Output CSV (default: results/cas_{sweep}.csv)")
    parser.add_argument("--n-bg", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS, help="Top-k values for CAS")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    global N_BG, N_TEST
    N_BG = args.n_bg
    N_TEST = args.n_test

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

    output_path = Path(args.output) if args.output else Path("results") / f"cas_{args.sweep}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["experiment", "modality", "model_type", "dev_f1"]
    for k in sorted(args.ks):
        fieldnames.append(f"cas@{k}")
    rows = []

    for exp_name, cp_path in tqdm(checkpoints, desc="CAS analysis"):
        raw = torch.load(cp_path, map_location="cpu", weights_only=False)
        cfg = raw.get("config")
        if not cfg:
            continue

        t_cfg = cfg.get("training", {})
        model_type = t_cfg.get("model_type", "tcn")
        if model_type in ("fusion", "fusion_tcn", "concat_fusion", "concat_fusion_tcn"):
            continue

        # Restore dev F1 from checkpoint metadata
        dev_f1 = raw.get("dev_f1", raw.get("best_dev_f1", 0.0))

        binning = cfg.get("binning")
        modality = resolve_modality(cfg.get("features"))
        feature_cols = get_feature_cols(cfg)
        clinical = CLINICAL_AUDIO_FEATURES if modality == "egemaps" else CLINICAL_VISUAL_FEATURES

        try:
            X_data, _, fc = load_data_for_shap(cfg, feature_cols, modality, binning)
            if len(X_data) == 0 or len(fc) == 0:
                continue

            model = build_model(cfg, len(fc), device).to(device)
            msd = raw.get("model_state_dict", raw.get("model_state", {}))
            model.load_state_dict(msd)
            model.eval()

            importance = run_shap_on_model(model, X_data, fc, device, model_type)
            if not importance:
                continue

            cas_scores = compute_cas_at_k(fc, importance, args.ks, clinical)
        except Exception as e:
            print(f"\n  SKIP {exp_name}: {e}")
            continue

        row = {
            "experiment": exp_name,
            "modality": modality,
            "model_type": model_type,
            "dev_f1": dev_f1,
        }
        for k, v in cas_scores.items():
            row[f"cas@{k}"] = round(v, 4)
        rows.append(row)

    if not rows:
        print("No results.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {output_path}")

    # Quick summary
    print(f"\n{'experiment':<30s} {'F1':>6s} {'CAS@5':>6s} {'CAS@10':>6s}")
    for r in sorted(rows, key=lambda r: r.get("cas@5", 0), reverse=True):
        print(f"{r['experiment']:<30s} {r['dev_f1']:>6.4f} {r.get('cas@5', 0):>6.4f} {r.get('cas@10', 0):>6.4f}")


if __name__ == "__main__":
    main()
