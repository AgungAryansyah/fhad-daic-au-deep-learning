import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_daic.config import get_feature_cols
from fhad_daic.data import (
    AUWindowDataset,
    FusionTCNDataset,
    apply_scaler,
    collate_fusion_tcn,
    fit_scaler,
    get_window_cache_path,
    load_fusion_tcn_sessions,
    load_sessions,
    resolve_label_mode,
    resolve_modality,
    slide_windows,
)
from fhad_daic.evaluate import find_latest_checkpoint, load_checkpoint, run_evaluation
from fhad_daic.models import FusionTCNModel, GRUModel, LSTMModel, MILTCN, TCN
from fhad_daic.training import evaluate_fusion_tcn, get_class_names


def _build_model_from_config(cfg: dict, num_inputs: int, device: torch.device, vis_dim: int | None = None, aud_dim: int | None = None):
    t_cfg = cfg["training"]
    model_type = t_cfg.get("model_type", "tcn")
    n_classes = t_cfg["num_classes"]

    if model_type == "fusion_tcn":
        ft_cfg = cfg["fusion_tcn"]
        return FusionTCNModel(
            vis_dim=vis_dim, aud_dim=aud_dim,
            vis_channels=ft_cfg["vis_channels"],
            aud_channels=ft_cfg["aud_channels"],
            kernel_size=ft_cfg["kernel_size"],
            tcn_dropout=ft_cfg["tcn_dropout"],
            fusion_hidden_dims=ft_cfg.get("fusion_hidden_dims", [64, 32]),
            fusion_dropout=ft_cfg.get("fusion_dropout", 0.7),
            num_classes=n_classes,
        ).to(device)
    elif model_type == "mil":
        tcn_cfg = cfg["tcn"]
        attn_dim = cfg.get("mil", {}).get("attn_dim", 64)
        return MILTCN(
            num_inputs=num_inputs, num_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"], dropout=tcn_cfg["dropout"],
            num_classes=n_classes, attn_dim=attn_dim,
        ).to(device)
    elif model_type == "gru":
        rnn_cfg = cfg["gru"]
        return GRUModel(
            num_inputs=num_inputs, hidden_size=rnn_cfg["hidden_size"],
            num_layers=rnn_cfg["num_layers"], dropout=rnn_cfg["dropout"],
            num_classes=n_classes, bidirectional=rnn_cfg.get("bidirectional", True),
        ).to(device)
    elif model_type == "lstm":
        rnn_cfg = cfg["lstm"]
        return LSTMModel(
            num_inputs=num_inputs, hidden_size=rnn_cfg["hidden_size"],
            num_layers=rnn_cfg["num_layers"], dropout=rnn_cfg["dropout"],
            num_classes=n_classes, bidirectional=rnn_cfg.get("bidirectional", True),
        ).to(device)
    else:
        tcn_cfg = cfg["tcn"]
        return TCN(
            num_inputs=num_inputs, num_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"], dropout=tcn_cfg["dropout"],
            num_classes=n_classes,
        ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint")
    parser.add_argument("--config", type=str, default="src/fhad_daic/config/visual/baseline.yaml",
                        help="Fallback YAML config if checkpoint lacks config")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (default: auto-detect latest)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name for checkpoint lookup (default: use global latest)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep name for checkpoint lookup")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = Path("checkpoints")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, sweep_name=args.sweep, experiment_name=args.experiment)
    print(f"Checkpoint: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = raw.get("config") or yaml.safe_load(open(args.config))
    print(f"Loading config: {cfg.get('experiment_name', args.config)}")

    t_cfg = cfg["training"]
    is_fusion_tcn = t_cfg.get("model_type") == "fusion_tcn"

    binning = cfg.get("binning")
    class_names = get_class_names(binning)

    if is_fusion_tcn:
        vis_cfg = cfg["features"]["visual"]
        vis_cols = vis_cfg.get("au_regression", []) + vis_cfg.get("au_binary", []) + vis_cfg.get("pose", [])
        if vis_cfg.get("include_confidence"):
            vis_cols = vis_cols + ["confidence"]
        if vis_cfg.get("confidence_aggregates"):
            vis_cols = vis_cols + ["confidence_mean", "confidence_std", "confidence_min"]
        aud_cols = cfg["features"]["audio"]["egemaps"]

        train_dir = Path(cfg["data"]["train_dir"])
        dev_dir = Path(cfg["data"]["dev_dir"])
        max_frames = cfg.get("data", {}).get("max_frames")

        print("Loading paired visual+audio sessions for evaluation...")
        train_vis, train_aud, train_y, _, _ = load_fusion_tcn_sessions(
            train_dir, vis_cols, aud_cols, binning=binning, max_frames=max_frames)
        dev_vis, dev_aud, dev_y, dev_aux_v, dev_aux_a = load_fusion_tcn_sessions(
            dev_dir, vis_cols, aud_cols, binning=binning, max_frames=max_frames)
        print(f"Dev sessions: {len(dev_vis)}")

        train_vis_sessions = [(X, int(y)) for X, y in zip(train_vis, train_y)]
        train_aud_sessions = [(X, int(y)) for X, y in zip(train_aud, train_y)]
        vis_scaler = fit_scaler(train_vis_sessions)
        aud_scaler = fit_scaler(train_aud_sessions)
        dev_vis = [vis_scaler.transform(X) for X in dev_vis]
        dev_aud = [aud_scaler.transform(X) for X in dev_aud]

        dev_ds = FusionTCNDataset(dev_vis, dev_aud, dev_y, dev_aux_v, dev_aux_a)
        dev_loader = DataLoader(dev_ds, batch_size=1, collate_fn=collate_fusion_tcn)

        model = _build_model_from_config(
            cfg, len(vis_cols), device, vis_dim=len(vis_cols), aud_dim=len(aud_cols))
        epoch = load_checkpoint(model, checkpoint_path, device)
        print(f"Loaded checkpoint from epoch {epoch}")

        _, f1, preds, labels = evaluate_fusion_tcn(model, dev_loader, nn.CrossEntropyLoss(), device)
        print(f"\nDev Macro F1: {f1:.4f}")
        print(classification_report(labels, preds, target_names=class_names, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, preds))
    else:
        feature_cols = get_feature_cols(cfg)
        label_mode = resolve_label_mode(binning)
        modality = resolve_modality(cfg.get("features"))

        w_cfg = cfg["windowing"]
        ws = w_cfg["window_size"]
        st = w_cfg["stride"]
        dev_pkl = get_window_cache_path("dev", ws, st, label_mode=label_mode, modality=modality)

        if dev_pkl.exists():
            print(f"Loading cached windowed data: {dev_pkl}")
            with open(dev_pkl, "rb") as f:
                dev_X, dev_y = pickle.load(f)
        else:
            print(f"Computing windowed data (window={ws}, stride={st})...")
            train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols, binning=binning, modality=modality)
            dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols, binning=binning, modality=modality)
            scaler = fit_scaler(train_sessions)
            dev_sessions = apply_scaler(dev_sessions, scaler)
            dev_X, dev_y = slide_windows(dev_sessions, ws, st)
            with open(dev_pkl, "wb") as f:
                pickle.dump((dev_X, dev_y), f)
            print(f"Saved windowed data to {dev_pkl}")

        dev_loader = DataLoader(AUWindowDataset(dev_X, dev_y), batch_size=t_cfg["batch_size"])
        model = _build_model_from_config(cfg, len(feature_cols), device)
        epoch = load_checkpoint(model, checkpoint_path, device)
        print(f"Loaded checkpoint from epoch {epoch}")

        run_evaluation(model, dev_loader, device, class_names=class_names)


if __name__ == "__main__":
    main()
