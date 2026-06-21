import argparse
import pickle
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_daic.config import get_feature_cols
from fhad_daic.data import AUWindowDataset, apply_scaler, fit_scaler, get_window_cache_path, load_sessions, resolve_label_mode, resolve_modality, slide_windows
from fhad_daic.evaluate import find_latest_checkpoint, load_checkpoint, run_evaluation
from fhad_daic.models import GRUModel, LSTMModel, MILTCN, TCN
from fhad_daic.training import get_class_names


def _build_model_from_config(cfg: dict, num_inputs: int, device: torch.device):
    t_cfg = cfg["training"]
    model_type = t_cfg.get("model_type", "tcn")
    n_classes = t_cfg["num_classes"]

    if model_type == "mil":
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

    feature_cols = get_feature_cols(cfg)
    binning = cfg.get("binning")
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

    t_cfg = cfg["training"]
    dev_loader = DataLoader(AUWindowDataset(dev_X, dev_y), batch_size=t_cfg["batch_size"])

    class_names = get_class_names(cfg.get("binning"))
    model = _build_model_from_config(cfg, len(feature_cols), device)
    epoch = load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {epoch}")

    run_evaluation(model, dev_loader, device, class_names=class_names)


if __name__ == "__main__":
    main()
