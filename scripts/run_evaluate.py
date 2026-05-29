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
from fhad_daic.data import AUWindowDataset, apply_scaler, fit_scaler, load_sessions, slide_windows
from fhad_daic.evaluate import load_checkpoint, run_evaluation
from fhad_daic.models import TCN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TCN for depression detection")
    parser.add_argument("--config", type=str, default="src/fhad_daic/config/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Loading config from: {args.config}")

    feature_cols = get_feature_cols(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dev_pkl = Path("windowed_dev.pkl")

    if dev_pkl.exists():
        print("Loading cached windowed data...")
        with open(dev_pkl, "rb") as f:
            dev_X, dev_y = pickle.load(f)
    else:
        print("Computing windowed data...")
        train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols)
        dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols)

        scaler = fit_scaler(train_sessions)
        dev_sessions = apply_scaler(dev_sessions, scaler)

        w_cfg = cfg["windowing"]
        dev_X, dev_y = slide_windows(dev_sessions, w_cfg["window_size"], w_cfg["stride"])

        with open(dev_pkl, "wb") as f:
            pickle.dump((dev_X, dev_y), f)
        print(f"Saved windowed data to {dev_pkl}")

    t_cfg = cfg["training"]
    dev_loader = DataLoader(AUWindowDataset(dev_X, dev_y), batch_size=t_cfg["batch_size"])

    tcn_cfg = cfg["tcn"]
    model = TCN(
        num_inputs=len(feature_cols),
        num_channels=tcn_cfg["num_channels"],
        kernel_size=tcn_cfg["kernel_size"],
        dropout=tcn_cfg["dropout"],
        num_classes=t_cfg["num_classes"],
    ).to(device)

    checkpoint_path = Path(cfg["data"]["checkpoints_dir"]) / "best.pt"
    epoch = load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {epoch}")

    run_evaluation(model, dev_loader, device, class_names=["not depressed", "depressed"])


if __name__ == "__main__":
    main()
