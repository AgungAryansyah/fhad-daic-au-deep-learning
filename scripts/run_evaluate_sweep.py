import argparse
import csv
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from fhad_daic.config import get_feature_cols
from fhad_daic.data import (
    AUWindowDataset,
    FusionDataset,
    FusionTCNDataset,
    MILWindowDataset,
    apply_scaler,
    collate_fusion_tcn,
    collate_mil,
    fit_scaler,
    get_window_cache_path,
    load_fusion_sessions,
    load_fusion_tcn_sessions,
    load_sessions,
    resolve_label_mode,
    resolve_modality,
    slide_windows,
)
from fhad_daic.evaluate import load_checkpoint, run_evaluation
from fhad_daic.models import ConcatFusionModel, FusionModel, FusionTCNModel, GRUModel, LSTMModel, MILTCN, TCN
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


def _evaluate_windowed(device, cfg, checkpoint_path, class_names):
    feature_cols = get_feature_cols(cfg)
    label_mode = resolve_label_mode(cfg.get("binning"))
    modality = resolve_modality(cfg.get("features"))
    t_cfg = cfg["training"]
    is_mil = t_cfg.get("model_type") == "mil"
    w_cfg = cfg["windowing"]
    ws = w_cfg["window_size"]
    st = w_cfg["stride"]
    mil = is_mil
    dev_pkl = get_window_cache_path("dev", ws, st, mil=mil, label_mode=label_mode, modality=modality)

    if dev_pkl.exists():
        with open(dev_pkl, "rb") as f:
            data = pickle.load(f)
    else:
        train_sessions = load_sessions(Path(cfg["data"]["train_dir"]), feature_cols, binning=cfg.get("binning"), modality=modality)
        dev_sessions = load_sessions(Path(cfg["data"]["dev_dir"]), feature_cols, binning=cfg.get("binning"), modality=modality)
        scaler = fit_scaler(train_sessions)
        dev_sessions = apply_scaler(dev_sessions, scaler)
        if is_mil:
            dev_X, dev_y, dev_sids = slide_windows(dev_sessions, ws, st, return_sids=True)
            data = (dev_X, dev_y, dev_sids)
        else:
            dev_X, dev_y = slide_windows(dev_sessions, ws, st)
            data = (dev_X, dev_y)
        dev_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(dev_pkl, "wb") as f:
            pickle.dump(data, f)

    model = _build_model_from_config(cfg, len(feature_cols), device)
    epoch = load_checkpoint(model, checkpoint_path, device)

    if is_mil:
        dev_X, dev_y, dev_sids = data
        dev_loader = DataLoader(MILWindowDataset(dev_X, dev_y, dev_sids), batch_size=t_cfg["batch_size"], collate_fn=collate_mil)
        from fhad_daic.training import evaluate_mil
        dev_loss, dev_f1, dev_auc, dev_preds, dev_labels = evaluate_mil(model, dev_loader, nn.CrossEntropyLoss(), device)
        from sklearn.metrics import classification_report
        report = classification_report(dev_labels, dev_preds, target_names=class_names, zero_division=0, output_dict=True)
        return epoch, {"predictions": dev_preds, "labels": dev_labels, "auc": dev_auc, "macro_f1": dev_f1, "report": report}
    else:
        dev_X, dev_y = data
        dev_loader = DataLoader(AUWindowDataset(dev_X, dev_y), batch_size=t_cfg["batch_size"])
        result = run_evaluation(model, dev_loader, device, class_names=class_names)
        return epoch, result


def _evaluate_fusion_functional(device, cfg, checkpoint_path, class_names):
    vis_cfg = cfg["features"]["visual"]
    vis_cols = vis_cfg.get("au_regression", []) + vis_cfg.get("au_binary", []) + vis_cfg.get("pose", [])
    aud_cols = cfg["features"]["audio"]["egemaps"]
    binning = cfg.get("binning")

    from fhad_daic.functional_features import extract_functional_features
    train_dir = Path(cfg["data"]["train_dir"])
    dev_dir = Path(cfg["data"]["dev_dir"])
    train_vis, train_aud, train_y, train_aux_v, train_aux_a = load_fusion_sessions(train_dir, vis_cols, aud_cols, binning=binning)
    dev_vis, dev_aud, dev_y, dev_aux_v, dev_aux_a = load_fusion_sessions(dev_dir, vis_cols, aud_cols, binning=binning)

    train_vis_sessions = [(X, int(y)) for X, y in zip(train_vis, train_y)]
    train_aud_sessions = [(X, int(y)) for X, y in zip(train_aud, train_y)]
    vis_scaler = fit_scaler(train_vis_sessions)
    aud_scaler = fit_scaler(train_aud_sessions)
    train_vis_sessions = apply_scaler(train_vis_sessions, vis_scaler)
    dev_vis_sessions = [(vis_scaler.transform(X), y) for X, y in zip(dev_vis, dev_y)]
    train_aud_sessions = apply_scaler(train_aud_sessions, aud_scaler)
    dev_aud_sessions = [(aud_scaler.transform(X), y) for X, y in zip(dev_aud, dev_y)]

    train_Fv, _ = extract_functional_features(train_vis_sessions)
    dev_Fv, _ = extract_functional_features(dev_vis_sessions)
    train_Fa, _ = extract_functional_features(train_aud_sessions)
    dev_Fa, _ = extract_functional_features(dev_aud_sessions)

    from fhad_daic.training import evaluate_fusion
    dev_loader = DataLoader(FusionDataset(dev_Fv, dev_Fa, dev_y, dev_aux_v, dev_aux_a), batch_size=len(dev_Fv))
    fusion_cfg = cfg.get("fusion", {})
    is_concat = cfg.get("training", {}).get("model_type") == "concat_fusion"
    if is_concat:
        model = ConcatFusionModel(vis_dim=dev_Fv.shape[1], aud_dim=dev_Fa.shape[1],
                                   hidden_dims=fusion_cfg.get("hidden_dims", [64, 32]),
                                   dropout=fusion_cfg.get("dropout", 0.7),
                                   num_classes=cfg["training"]["num_classes"]).to(device)
    else:
        model = FusionModel(vis_dim=dev_Fv.shape[1], aud_dim=dev_Fa.shape[1],
                             hidden_dims=fusion_cfg.get("hidden_dims", [64, 32]),
                             dropout=fusion_cfg.get("dropout", 0.7),
                             num_classes=cfg["training"]["num_classes"]).to(device)
    epoch = load_checkpoint(model, checkpoint_path, device)
    dev_loss, dev_f1, dev_auc, dev_preds, dev_labels = evaluate_fusion(model, dev_loader, nn.CrossEntropyLoss(), device)
    from sklearn.metrics import classification_report
    report = classification_report(dev_labels, dev_preds, target_names=class_names, zero_division=0, output_dict=True)
    return epoch, {"predictions": dev_preds, "labels": dev_labels, "auc": dev_auc, "macro_f1": dev_f1, "report": report}


def _evaluate_fusion_tcn(device, cfg, checkpoint_path, class_names):
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
    binning = cfg.get("binning")

    train_vis, train_aud, train_y, _, _ = load_fusion_tcn_sessions(
        train_dir, vis_cols, aud_cols, binning=binning, max_frames=max_frames)
    dev_vis, dev_aud, dev_y, dev_aux_v, dev_aux_a = load_fusion_tcn_sessions(
        dev_dir, vis_cols, aud_cols, binning=binning, max_frames=max_frames)

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

    from fhad_daic.training import evaluate_fusion_tcn as ef
    _, f1, auc, preds, labels = ef(model, dev_loader, nn.CrossEntropyLoss(), device)

    report = classification_report(labels, preds, target_names=class_names, zero_division=0, output_dict=True)
    return epoch, {"predictions": preds, "labels": labels, "auc": auc, "macro_f1": f1, "report": report}


def list_sweep_experiments(sweep_dir: Path) -> list[tuple[Path, Path]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate all experiments in a sweep")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep name in checkpoints/ dir")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: results/{sweep}.csv)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sweep_dir = Path("checkpoints") / args.sweep
    if not sweep_dir.exists():
        print(f"Sweep not found: {sweep_dir}")
        return

    experiments = list_sweep_experiments(sweep_dir)
    if not experiments:
        print(f"No experiments with valid checkpoints found in {sweep_dir}")
        return
    print(f"Found {len(experiments)} experiment(s)")

    output_path = Path(args.output) if args.output else Path("results") / f"{args.sweep}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    fieldnames = ["experiment", "model_type", "modality", "epoch", "macro_f1", "auc"]
    labels_set = set()

    for exp_name, cp_path in experiments:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"Checkpoint: {cp_path}")

        raw = torch.load(cp_path, map_location=device, weights_only=False)
        cfg = raw.get("config")
        if not cfg:
            print("  SKIP: checkpoint has no config")
            continue

        t_cfg = cfg.get("training", {})
        model_type = t_cfg.get("model_type", "tcn")
        is_fusion_tcn = model_type in ("fusion_tcn", "concat_fusion_tcn")
        is_fusion_func = model_type in ("fusion", "concat_fusion")
        binning = cfg.get("binning")
        class_names = get_class_names(binning)

        try:
            if is_fusion_tcn:
                epoch, result = _evaluate_fusion_tcn(device, cfg, cp_path, class_names)
            elif is_fusion_func:
                epoch, result = _evaluate_fusion_functional(device, cfg, cp_path, class_names)
            else:
                epoch, result = _evaluate_windowed(device, cfg, cp_path, class_names)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        modality = "egemaps" if "egemaps" in str(cfg.get("features", {}).get("modality", "au")) else "visual"
        if is_fusion_tcn:
            modality = "fusion-tcn"

        row = {
            "experiment": exp_name,
            "model_type": model_type,
            "modality": modality,
            "epoch": epoch,
            "macro_f1": result.get("macro_f1", result.get("auc", 0)),
            "auc": result.get("auc", 0),
        }

        report = result.get("report", {})
        for cls_name, metrics in report.items():
            if cls_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            labels_set.add(cls_name)
            row[f"{cls_name}_f1"] = metrics.get("f1-score", 0)
            row[f"{cls_name}_precision"] = metrics.get("precision", 0)
            row[f"{cls_name}_recall"] = metrics.get("recall", 0)

        rows.append(row)
        print(f"  Epoch: {epoch}  F1: {row['macro_f1']:.4f}  AUC: {row['auc']:.4f}")

    if not rows:
        print("\nNo results to save.")
        return

    for label in sorted(labels_set):
        fieldnames.extend([f"{label}_f1", f"{label}_precision", f"{label}_recall"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path}")
    print(f"  {len(rows)} experiments evaluated")


if __name__ == "__main__":
    main()
