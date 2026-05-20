import argparse
import pickle
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_tcn.config import get_feature_cols
from fhad_tcn.dataset import (
    AUWindowDataset,
    apply_scaler,
    compute_class_weights,
    fit_scaler,
    load_sessions,
    slide_windows,
)
from fhad_tcn.model import TCN
from fhad_tcn.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN for depression detection")
    parser.add_argument("--config", type=str, default="src/fhad_tcn/config/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Loading config from: {args.config}")

    feature_cols = get_feature_cols(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_pkl = Path("windowed_train.pkl")
    dev_pkl = Path("windowed_dev.pkl")

    if train_pkl.exists() and dev_pkl.exists():
        print("Loading cached windowed data...")
        with open(train_pkl, "rb") as f:
            train_X, train_y = pickle.load(f)
        with open(dev_pkl, "rb") as f:
            dev_X, dev_y = pickle.load(f)
    else:
        print("Computing windowed data...")
        train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols)
        dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols)

        scaler = fit_scaler(train_sessions)
        train_sessions = apply_scaler(train_sessions, scaler)
        dev_sessions = apply_scaler(dev_sessions, scaler)

        w_cfg = cfg["windowing"]
        train_X, train_y = slide_windows(train_sessions, w_cfg["window_size"], w_cfg["stride"])
        dev_X, dev_y = slide_windows(dev_sessions, w_cfg["window_size"], w_cfg["stride"])

        with open(train_pkl, "wb") as f:
            pickle.dump((train_X, train_y), f)
        with open(dev_pkl, "wb") as f:
            pickle.dump((dev_X, dev_y), f)
        print(f"Saved windowed data to {train_pkl} and {dev_pkl}")


    t_cfg = cfg["training"]
    num_workers = t_cfg.get("num_workers", 0)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        AUWindowDataset(train_X, train_y),
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dev_loader = DataLoader(
        AUWindowDataset(dev_X, dev_y),
        batch_size=t_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    tcn_cfg = cfg["tcn"]
    model = TCN(
        num_inputs=len(feature_cols),
        num_channels=tcn_cfg["num_channels"],
        kernel_size=tcn_cfg["kernel_size"],
        dropout=tcn_cfg["dropout"],
        num_classes=t_cfg["num_classes"],
    ).to(device)

    class_weights = compute_class_weights(train_y, t_cfg["num_classes"]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])

    result = run_training(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=t_cfg["num_epochs"],
        patience=t_cfg["early_stopping_patience"],
        checkpoint_path=Path(cfg["data"]["checkpoints_dir"]) / "best.pt",
        device=device,
        config=cfg,
    )

    print(f"\nBest dev macro F1: {result['best_dev_f1']:.4f}")


if __name__ == "__main__":
    main()
