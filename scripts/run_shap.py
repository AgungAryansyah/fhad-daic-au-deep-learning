import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

matplotlib.use("Agg")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_daic.config import get_feature_cols
from fhad_daic.data import (
    AUWindowDataset,
    FusionDataset,
    FusionTCNDataset,
    apply_scaler,
    collate_fusion_tcn,
    fit_scaler,
    load_fusion_sessions,
    load_fusion_tcn_sessions,
    load_sessions,
    resolve_label_mode,
    resolve_modality,
    slide_windows,
)
from fhad_daic.functional_features import extract_functional_features
from fhad_daic.models import FusionModel, FusionTCNModel, GRUModel, LSTMModel, MLP, MILTCN, TCN
from fhad_daic.training.utils import load_full_checkpoint


DEFAULT_SHAP_CFG = {
    "mode": "kernel",
    "output_dir": "shap_output",
    "num_background": 100,
    "num_test": 20,
}


def _merge_config(shap_cfg: dict, args: argparse.Namespace) -> dict:
    merged = {**DEFAULT_SHAP_CFG, **shap_cfg}

    if args.mode:
        merged["mode"] = args.mode
    if args.checkpoint:
        merged["checkpoint"] = args.checkpoint
    if args.config:
        merged["train_config_path"] = args.config
    if args.num_bg is not None:
        merged["num_background"] = args.num_bg
    if args.num_test is not None:
        merged["num_test"] = args.num_test
    if args.output_dir:
        merged["output_dir"] = args.output_dir

    model_cfg = merged.get("model", {})
    if "checkpoint" not in merged:
        merged["checkpoint"] = model_cfg.get("checkpoint")
    if "train_config_path" not in merged:
        merged["train_config_path"] = model_cfg.get("config")
    if "num_background" not in merged:
        merged["num_background"] = merged.get("kernel", {}).get("num_background", 100)
    if "num_test" not in merged:
        merged["num_test"] = merged.get("kernel", {}).get("num_test", 20)

    return merged


def _build_model(cfg: dict, num_inputs: int, device: torch.device):
    t_cfg = cfg["training"]
    mt = t_cfg.get("model_type", "tcn")
    nc = t_cfg["num_classes"]

    if mt == "fusion_tcn":
        ft_cfg = cfg["fusion_tcn"]
        vis_dim = num_inputs // 2
        aud_dim = num_inputs - vis_dim
        return FusionTCNModel(
            vis_dim=vis_dim, aud_dim=aud_dim,
            vis_channels=ft_cfg["vis_channels"],
            aud_channels=ft_cfg["aud_channels"],
            kernel_size=ft_cfg["kernel_size"],
            tcn_dropout=ft_cfg["tcn_dropout"],
            fusion_hidden_dims=ft_cfg["fusion_hidden_dims"],
            fusion_dropout=ft_cfg["fusion_dropout"],
            num_classes=nc,
        ).to(device)
    elif mt == "mil":
        tc = cfg["tcn"]
        ad = cfg.get("mil", {}).get("attn_dim", 64)
        return MILTCN(num_inputs, tc["num_channels"], tc["kernel_size"], tc["dropout"], nc, ad).to(device)
    elif mt == "gru":
        rc = cfg["gru"]
        return GRUModel(num_inputs, rc["hidden_size"], rc["num_layers"], rc["dropout"], nc, rc.get("bidirectional", True)).to(device)
    elif mt == "lstm":
        rc = cfg["lstm"]
        return LSTMModel(num_inputs, rc["hidden_size"], rc["num_layers"], rc["dropout"], nc, rc.get("bidirectional", True)).to(device)
    elif mt == "functional":
        fc = cfg.get("mlp", {})
        return MLP(num_inputs, fc.get("hidden_dims", [64, 32]), fc.get("dropout", 0.7), nc).to(device)
    elif mt == "fusion":
        vd = num_inputs // 2
        ad = num_inputs - vd
        fc = cfg.get("fusion", {})
        return FusionModel(vd, ad, fc.get("hidden_dims", [64, 32]), fc.get("dropout", 0.7), nc).to(device)
    else:
        tc = cfg["tcn"]
        return TCN(num_inputs, tc["num_channels"], tc["kernel_size"], tc["dropout"], nc).to(device)


def _load_functional_data(cfg, device, n_samples=200):
    feature_cols = get_feature_cols(cfg)
    binning = cfg.get("binning")
    modality = resolve_modality(cfg.get("features"))

    train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols, binning=binning, modality=modality)
    scaler = fit_scaler(train_sessions)
    train_sessions = apply_scaler(train_sessions, scaler)

    X, y = extract_functional_features(train_sessions)
    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    bg = torch.from_numpy(X[idx]).to(device)
    return bg, feature_cols


def _load_fusion_data(cfg, device, n_samples=200):
    vis_cfg = cfg["features"]["visual"]
    vis_cols = vis_cfg.get("au_regression", []) + vis_cfg.get("au_binary", []) + vis_cfg.get("pose", [])
    aud_cols = cfg["features"]["audio"]["egemaps"]
    binning = cfg.get("binning")

    train_vis, train_aud, train_y, aux_v, aux_a = load_fusion_sessions(
        Path(cfg["data"]["train_dir"]), vis_cols, aud_cols, binning=binning)

    train_vis_s = [(X, int(y)) for X, y in zip(train_vis, train_y)]
    train_aud_s = [(X, int(y)) for X, y in zip(train_aud, train_y)]
    vis_s = fit_scaler(train_vis_s)
    aud_s = fit_scaler(train_aud_s)
    train_vis_s = apply_scaler(train_vis_s, vis_s)
    train_aud_s = apply_scaler(train_aud_s, aud_s)

    Fv, _ = extract_functional_features(train_vis_s)
    Fa, _ = extract_functional_features(train_aud_s)

    n = min(n_samples, len(Fv))
    idx = np.random.choice(len(Fv), size=n, replace=False)

    bg = torch.from_numpy(np.concatenate([Fv[idx], Fa[idx]], axis=1)).to(device)
    feature_names = [f"{c}_v" for c in vis_cols] + [f"{c}_a" for c in aud_cols]
    return bg, feature_names


def _load_temporal_data(cfg, device, n_samples=200):
    feature_cols = get_feature_cols(cfg)
    binning = cfg.get("binning")
    modality = resolve_modality(cfg.get("features"))

    train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols, binning=binning, modality=modality)
    scaler = fit_scaler(train_sessions)
    train_sessions = apply_scaler(train_sessions, scaler)

    w_cfg = cfg["windowing"]
    X, _ = slide_windows(train_sessions, w_cfg["window_size"], w_cfg["stride"])
    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    bg = torch.from_numpy(X[idx]).to(device)
    return bg, feature_cols


def _load_fusion_tcn_shap_data(cfg, device, checkpoint_path, n_samples=200):
    vis_cfg = cfg["features"]["visual"]
    vis_cols = vis_cfg.get("au_regression", []) + vis_cfg.get("au_binary", []) + vis_cfg.get("pose", [])
    if vis_cfg.get("include_confidence"):
        vis_cols = vis_cols + ["confidence"]
    if vis_cfg.get("confidence_aggregates"):
        vis_cols = vis_cols + ["confidence_mean", "confidence_std", "confidence_min"]
    aud_cols = cfg["features"]["audio"]["egemaps"]
    max_frames = cfg.get("data", {}).get("max_frames")
    binning = cfg.get("binning")

    train_vis, train_aud, train_y, aux_v, aux_a = load_fusion_tcn_sessions(
        Path(cfg["data"]["train_dir"]), vis_cols, aud_cols, binning=binning, max_frames=max_frames)

    train_vis_s = [(X, int(y)) for X, y in zip(train_vis, train_y)]
    train_aud_s = [(X, int(y)) for X, y in zip(train_aud, train_y)]
    vis_scaler = fit_scaler(train_vis_s)
    aud_scaler = fit_scaler(train_aud_s)

    ft_cfg = cfg["fusion_tcn"]
    full_model = FusionTCNModel(
        vis_dim=len(vis_cols), aud_dim=len(aud_cols),
        vis_channels=ft_cfg["vis_channels"], aud_channels=ft_cfg["aud_channels"],
        kernel_size=ft_cfg["kernel_size"], tcn_dropout=ft_cfg["tcn_dropout"],
        fusion_hidden_dims=ft_cfg.get("fusion_hidden_dims", [64, 32]),
        fusion_dropout=ft_cfg.get("fusion_dropout", 0.7),
        num_classes=cfg["training"]["num_classes"],
    ).to(device)
    load_full_checkpoint(checkpoint_path, full_model, device=device)
    full_model.eval()

    fusion_features = []
    with torch.no_grad():
        for i in range(len(train_vis)):
            X_v = torch.from_numpy(vis_scaler.transform(train_vis[i])).unsqueeze(0).to(device)
            X_a = torch.from_numpy(aud_scaler.transform(train_aud[i])).unsqueeze(0).to(device)
            mask_v = torch.ones(1, X_v.size(1), dtype=torch.bool, device=device)
            mask_a = torch.ones(1, X_a.size(1), dtype=torch.bool, device=device)
            av = {k: torch.tensor([v[i]], device=device) for k, v in aux_v.items()}
            aa = {k: torch.tensor([v[i]], device=device) for k, v in aux_a.items()}

            V = full_model.proj_v(full_model.vis_encoder(X_v, mask_v))
            A = full_model.proj_a(full_model.aud_encoder(X_a, mask_a))
            w_v, w_a = full_model.reliability(av, aa)
            F = torch.cat([w_v * V, w_a * A], dim=-1)
            fusion_features.append(F.cpu().numpy())

    F_all = np.concatenate(fusion_features, axis=0)
    n = min(n_samples, len(F_all))
    idx = np.random.choice(len(F_all), size=n, replace=False)
    bg = torch.from_numpy(F_all[idx]).to(device)

    feature_names = (
        [f"vis_emb_{i}" for i in range(F_all.shape[1] // 2)] +
        [f"aud_emb_{i}" for i in range(F_all.shape[1] // 2, F_all.shape[1])]
    )

    classifier = full_model.classifier
    wrapped = torch.nn.Sequential(classifier).to(device)
    return bg, feature_names, wrapped


def run_shap(shap_cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(shap_cfg["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Checkpoint: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_cfg = raw.get("config")
    if train_cfg is None:
        train_cfg_path = shap_cfg.get("train_config_path")
        if train_cfg_path:
            with open(train_cfg_path) as f:
                train_cfg = yaml.safe_load(f)
        else:
            raise ValueError("Checkpoint lacks config and no --config fallback provided")
    print(f"Experiment: {train_cfg.get('experiment_name', 'unknown')}")

    mode = shap_cfg.get("mode", "kernel")
    print(f"SHAP mode: {mode}")

    if mode == "timeshap":
        _run_timeshap(train_cfg, shap_cfg, device, checkpoint_path)
        return

    mt = train_cfg["training"].get("model_type", "tcn")
    is_temporal = mt in ("tcn", "gru", "lstm")
    is_functional = mt == "functional"
    is_fusion = mt == "fusion"
    is_fusion_tcn = mt == "fusion_tcn"

    n_bg = shap_cfg["num_background"]
    n_test = shap_cfg["num_test"]

    if is_fusion_tcn:
        bg_data, feature_names, model = _load_fusion_tcn_shap_data(train_cfg, device, checkpoint_path, n_bg)
        is_temporal = False
    elif is_fusion:
        bg_data, feature_names = _load_fusion_data(train_cfg, device, n_bg)
    elif is_functional:
        bg_data, feature_names = _load_functional_data(train_cfg, device, n_bg)
    else:
        bg_data, feature_names = _load_temporal_data(train_cfg, device, n_bg)

    if not is_fusion_tcn:
        num_inputs = bg_data.shape[1] if (is_functional or is_fusion) else bg_data.shape[2]
        model = _build_model(train_cfg, num_inputs, device)
        load_full_checkpoint(checkpoint_path, model, device=device)
        model.eval()

    if is_temporal:
        _run_temporal_shap(model, bg_data, feature_names, train_cfg, shap_cfg)
    else:
        _run_tabular_shap(model, bg_data, feature_names, train_cfg, shap_cfg)


def _run_timeshap(train_cfg: dict, shap_cfg: dict, device: torch.device, checkpoint_path: Path):
    ts_cfg = shap_cfg.get("timeshap", {})
    modality = ts_cfg.get("modality", "both")
    num_sessions = ts_cfg.get("num_sessions", 5)
    baseline = ts_cfg.get("baseline", "zeros")

    output_dir = Path(shap_cfg["output_dir"]) / train_cfg.get("experiment_name", "shap") / "timeshap"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"timeshap: modality={modality}, sessions={num_sessions}, baseline={baseline}")
    print(f"timeshap support not yet implemented — output dir: {output_dir}")


def _extract_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _run_tabular_shap(model, bg_data, feature_names, train_cfg, shap_cfg):
    import shap

    bg_np = bg_data.cpu().numpy()
    n_bg = min(shap_cfg["num_background"], len(bg_np))
    bg_sample = bg_np[np.random.choice(len(bg_np), size=n_bg, replace=False)]

    print(f"Background: {bg_sample.shape}  Feature names: {len(feature_names)}")

    model_device = next(model.parameters()).device

    def _predict(x_np):
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(model_device)
        with torch.no_grad():
            return model(x_t).cpu().numpy()

    explainer = shap.KernelExplainer(_predict, bg_sample, seed=42)
    n_test = min(shap_cfg["num_test"], len(bg_np))
    test_sample = bg_np[np.random.choice(len(bg_np), size=n_test, replace=False)]
    shap_values = explainer.shap_values(test_sample, nsamples=200)
    sv = _extract_shap_values(shap_values)

    output_dir = Path(shap_cfg["output_dir"]) / train_cfg.get("experiment_name", "shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_summary(sv, test_sample, feature_names, output_dir)
    _plot_bar(sv, output_dir)
    _plot_waterfall(sv, test_sample, feature_names, output_dir)

    np.save(output_dir / "shap_values.npy", sv)
    np.save(output_dir / "test_data.npy", test_sample)
    print(f"Saved to {output_dir}")


def _run_temporal_shap(model, bg_data, feature_names, train_cfg, shap_cfg):
    import shap

    bg_np = bg_data.cpu().numpy()
    n_bg = min(shap_cfg["num_background"], len(bg_np))
    bg_sample = bg_np[np.random.choice(len(bg_np), size=n_bg, replace=False)]

    print(f"Background: {bg_sample.shape}  Features: {len(feature_names)}")

    bg_t = torch.from_numpy(bg_sample).float()
    explainer = shap.GradientExplainer(model, bg_t)
    n_test = min(shap_cfg["num_test"], len(bg_np))
    test_sample = bg_np[np.random.choice(len(bg_np), size=n_test, replace=False)]
    test_t = torch.from_numpy(test_sample).float()
    shap_values = explainer.shap_values(test_t)
    sv = _extract_shap_values(shap_values)

    output_dir = Path(shap_cfg["output_dir"]) / train_cfg.get("experiment_name", "shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    sv_agg = np.abs(sv).mean(axis=1)

    _plot_bar(sv_agg, output_dir, suffix="_temporal")
    _plot_heatmap(sv, feature_names, output_dir)

    np.save(output_dir / "shap_values.npy", sv)
    np.save(output_dir / "shap_aggregated.npy", sv_agg)
    print(f"Saved to {output_dir}")


def _plot_summary(sv, test_data, feature_names, output_dir):
    import shap
    top_n = min(20, len(feature_names))
    idx = np.argsort(np.abs(sv).mean(axis=0))[-top_n:]
    idx_list = idx.tolist()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv[:, idx_list], test_data[:, idx_list], feature_names=[feature_names[i] for i in idx_list], show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_bar(sv, output_dir, suffix=""):
    import shap
    plt.figure(figsize=(10, 6))
    if len(sv.shape) == 1:
        idx = np.argsort(np.abs(sv))[-20:]
        plt.barh(range(20), sv[idx])
        plt.yticks(range(20), [str(i) for i in idx])
    else:
        shap.summary_plot(sv, show=False, plot_type="bar", max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"summary_bar{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_waterfall(sv, test_data, feature_names, output_dir):
    import shap
    top_n = min(20, len(feature_names))
    idx = np.argsort(np.abs(sv).mean(axis=0))[-top_n:]
    idx_sorted = idx[np.argsort(sv[0, idx])]
    idx_sorted_list = idx_sorted.tolist()
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv[0, idx_sorted_list],
            base_values=float(sv[0].sum()),
            data=test_data[0, idx_sorted_list],
            feature_names=[feature_names[i] for i in idx_sorted_list],
        ),
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "waterfall_sample0.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_heatmap(sv, feature_names, output_dir):
    import shap
    sv_sample = sv[0]
    top_n = min(20, len(feature_names))
    idx = np.argsort(np.abs(sv_sample).mean(axis=0))[-top_n:]
    idx_list = idx.tolist()
    plt.figure(figsize=(14, 8))
    shap.heatmap_plot(
        shap.Explanation(
            values=sv_sample[:, idx_list],
            feature_names=[feature_names[i] for i in idx_list],
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_sample0.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for fhad-daic models")
    parser.add_argument("--shap-config", type=str, required=True, help="Path to SHAP config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Override fallback training config path")
    parser.add_argument("--num_bg", type=int, default=None, help="Override number of background samples")
    parser.add_argument("--num_test", type=int, default=None, help="Override number of test samples")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--mode", type=str, default=None, help="Override SHAP mode (kernel|gradient|timeshap)")
    args = parser.parse_args()

    with open(args.shap_config) as f:
        shap_cfg = yaml.safe_load(f)
    shap_cfg["_config_path"] = args.shap_config
    print(f"Loading SHAP config from: {args.shap_config}")

    shap_cfg = _merge_config(shap_cfg, args)
    run_shap(shap_cfg)


if __name__ == "__main__":
    main()
