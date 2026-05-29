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
from fhad_tcn.cross_validation import run_loso_cv_mlp
from fhad_tcn.dataset import (
    AUWindowDataset,
    MILWindowDataset,
    apply_scaler,
    collate_mil,
    compute_class_weights,
    fit_scaler,
    load_sessions,
    slide_windows,
)
from fhad_tcn.functional_features import extract_functional_features
from fhad_tcn.model import MLP, MILTCN, TCN
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
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    is_mil = cfg.get("training", {}).get("model_type") == "mil"
    is_functional = cfg.get("training", {}).get("model_type") == "functional"

    t_cfg = cfg["training"]

    cv_mode = t_cfg.get("cv_mode")
    if cv_mode == "loso":
        result = run_loso_cv_mlp(
            train_dir=Path(cfg["data"]["train_dir"]),
            dev_dir=Path(cfg["data"]["dev_dir"]),
            feature_cols=feature_cols,
            cfg=cfg,
            device=device,
        )
        print(f"\nLOSO CV macro F1: {result['macro_f1']:.4f}")
        return

    if is_functional:
        train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols)
        dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols)

        scaler = fit_scaler(train_sessions)
        train_sessions = apply_scaler(train_sessions, scaler)
        dev_sessions = apply_scaler(dev_sessions, scaler)

        print(f"Extracting functional features from {len(train_sessions)} training sessions...")
        train_X, train_y = extract_functional_features(train_sessions)
        dev_X, dev_y = extract_functional_features(dev_sessions)
        print(f"Functional features shape: {train_X.shape[1]}")

        train_loader = DataLoader(
            AUWindowDataset(train_X, train_y),
            batch_size=len(train_X),
            shuffle=True,
        )
        dev_loader = DataLoader(
            AUWindowDataset(dev_X, dev_y),
            batch_size=len(dev_X),
        )

        mlp_cfg = cfg.get("mlp", {})
        model = MLP(
            num_features=train_X.shape[1],
            hidden_dims=mlp_cfg.get("hidden_dims", [64, 32]),
            dropout=mlp_cfg.get("dropout", 0.7),
            num_classes=t_cfg["num_classes"],
        ).to(device)

        class_weights = compute_class_weights(train_y, t_cfg["num_classes"]).to(device)
        label_smoothing = t_cfg.get("label_smoothing", 0.0)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=mlp_cfg.get("weight_decay", 0.01),
        )

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
        return

    if is_mil:
        train_pkl = Path("windowed_train_mil.pkl")
        dev_pkl = Path("windowed_dev_mil.pkl")
    else:
        train_pkl = Path("windowed_train.pkl")
        dev_pkl = Path("windowed_dev.pkl")

    if train_pkl.exists() and dev_pkl.exists():
        print("Loading cached windowed data...")
        with open(train_pkl, "rb") as f:
            train_data = pickle.load(f)
        with open(dev_pkl, "rb") as f:
            dev_data = pickle.load(f)
        if is_mil:
            train_X, train_y, train_sids = train_data
            dev_X, dev_y, dev_sids = dev_data
        else:
            train_X, train_y = train_data
            dev_X, dev_y = dev_data
    else:
        print("Computing windowed data...")
        train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols)
        dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols)

        scaler = fit_scaler(train_sessions)
        train_sessions = apply_scaler(train_sessions, scaler)
        dev_sessions = apply_scaler(dev_sessions, scaler)

        w_cfg = cfg["windowing"]
        if is_mil:
            train_X, train_y, train_sids = slide_windows(
                train_sessions, w_cfg["window_size"], w_cfg["stride"], return_sids=True
            )
            dev_X, dev_y, dev_sids = slide_windows(
                dev_sessions, w_cfg["window_size"], w_cfg["stride"], return_sids=True
            )
            train_data = (train_X, train_y, train_sids)
            dev_data = (dev_X, dev_y, dev_sids)
        else:
            train_X, train_y = slide_windows(train_sessions, w_cfg["window_size"], w_cfg["stride"])
            dev_X, dev_y = slide_windows(dev_sessions, w_cfg["window_size"], w_cfg["stride"])
            train_data = (train_X, train_y)
            dev_data = (dev_X, dev_y)

        with open(train_pkl, "wb") as f:
            pickle.dump(train_data, f)
        with open(dev_pkl, "wb") as f:
            pickle.dump(dev_data, f)
        print(f"Saved windowed data to {train_pkl} and {dev_pkl}")

    num_workers = t_cfg.get("num_workers", 0)
    pin_memory = device.type == "cuda"

    if is_mil:
        train_loader = DataLoader(
            MILWindowDataset(train_X, train_y, train_sids),
            batch_size=t_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_mil,
        )
        dev_loader = DataLoader(
            MILWindowDataset(dev_X, dev_y, dev_sids),
            batch_size=t_cfg["batch_size"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_mil,
        )
    else:
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
    if is_mil:
        attn_dim = cfg.get("mil", {}).get("attn_dim", 64)
        model = MILTCN(
            num_inputs=len(feature_cols),
            num_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
            num_classes=t_cfg["num_classes"],
            attn_dim=attn_dim,
        ).to(device)
    else:
        model = TCN(
            num_inputs=len(feature_cols),
            num_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
            num_classes=t_cfg["num_classes"],
        ).to(device)

    class_weights = compute_class_weights(train_y, t_cfg["num_classes"]).to(device)
    label_smoothing = t_cfg.get("label_smoothing", 0.0)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=tcn_cfg.get("weight_decay", 0.0),
    )

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
        is_mil=is_mil,
    )

    print(f"\nBest dev macro F1: {result['best_dev_f1']:.4f}")


if __name__ == "__main__":
    main()
