import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DEGRADATION_LEVELS = {
    "L1": {
        "label": "mild",
        "vis": {"drop_fraction": 0.20, "au_noise_std": 0.1, "pose_noise_std": 0.05},
        "aud": {"noise_std": 0.1, "noise_fraction": 0.30, "zero_hnr": False},
    },
    "L2": {
        "label": "moderate",
        "vis": {"drop_fraction": 0.40, "au_noise_std": 0.5, "pose_noise_std": 0.2},
        "aud": {"noise_std": 0.3, "noise_fraction": 0.60, "zero_hnr": False},
    },
    "L3": {
        "label": "severe",
        "vis": {"drop_fraction": 0.60, "au_noise_std": 1.0, "pose_noise_std": 0.5},
        "aud": {"noise_std": 0.5, "noise_fraction": 0.90, "zero_hnr": True},
    },
}

VIS_METADATA = ["participant_id", "phq_score", "phq_binary", "frame", "timestamp"]
AUD_METADATA = ["participant_id", "phq_score", "phq_binary", "frameTime"]


def _is_au_col(col: str) -> bool:
    return col.startswith("AU")


def _is_pose_col(col: str) -> bool:
    return col.startswith("pose_")


def degrade_visual(df: pd.DataFrame, level: dict) -> pd.DataFrame:
    cfg = level["vis"]
    n = len(df)
    n_keep = max(1, int(n * (1.0 - cfg["drop_fraction"])))
    keep_idx = sorted(np.random.choice(n, n_keep, replace=False))
    df = df.iloc[keep_idx].copy()

    feature_cols = [c for c in df.columns if c not in VIS_METADATA]
    rng = np.random.RandomState(42)
    for c in feature_cols:
        if _is_au_col(c) and cfg["au_noise_std"] > 0:
            df[c] = df[c].astype(float) + rng.normal(0, cfg["au_noise_std"], len(df))
        elif _is_pose_col(c) and cfg["pose_noise_std"] > 0:
            df[c] = df[c].astype(float) + rng.normal(0, cfg["pose_noise_std"], len(df))
    return df


def degrade_audio(df: pd.DataFrame, level: dict) -> pd.DataFrame:
    cfg = level["aud"]
    feature_cols = [c for c in df.columns if c not in AUD_METADATA and c not in ("participant_id", "phq_score", "phq_binary", "frameTime")]
    rng = np.random.RandomState(42)
    n_cols = len(feature_cols)
    n_noise = max(1, int(n_cols * cfg["noise_fraction"]))
    noise_cols = list(rng.choice(feature_cols, n_noise, replace=False))
    for c in noise_cols:
        if cfg["zero_hnr"] and c == "HNRdBACF_sma3nz":
            df[c] = 0.0
        elif cfg["noise_std"] > 0:
            df[c] = df[c].astype(float) + rng.normal(0, cfg["noise_std"], len(df))
    return df


def generate_degraded(source_dir: Path, output_root: Path, level_name: str) -> None:
    level_cfg = DEGRADATION_LEVELS[level_name]
    for split in ["train", "dev"]:
        src_split = source_dir / split
        if not src_split.exists():
            continue
        dst_split = output_root / level_name / split
        dst_split.mkdir(parents=True, exist_ok=True)

        vis_files = sorted(src_split.glob("*_clean.csv"))
        for vf in vis_files:
            sid = vf.stem.replace("_clean", "")
            vdf = pd.read_csv(vf)
            if vdf.empty:
                continue
            vdf = degrade_visual(vdf, level_cfg)
            vdf.to_csv(dst_split / f"{sid}_clean.csv", index=False)

        aud_files = sorted(src_split.glob("*_egemaps_clean.csv"))
        for af in aud_files:
            sid = af.stem.replace("_egemaps_clean", "")
            adf = pd.read_csv(af)
            if adf.empty:
                continue
            adf = degrade_audio(adf, level_cfg)
            adf.to_csv(dst_split / f"{sid}_egemaps_clean.csv", index=False)

    print(f"  {level_name} ({level_cfg['label']}): vis drop {level_cfg['vis']['drop_fraction']:.0%}, "
          f"au σ={level_cfg['vis']['au_noise_std']}, aud σ={level_cfg['aud']['noise_std']}")


def main():
    parser = argparse.ArgumentParser(description="Generate degraded copies of cleaned CSVs for fusion testing")
    parser.add_argument("--source", type=Path, default=Path("data"),
                        help="Source data directory with train/ and dev/ subdirs")
    parser.add_argument("--output", type=Path, default=Path("data_degraded"),
                        help="Output root directory")
    parser.add_argument("--levels", nargs="*", default=["L1", "L2", "L3"],
                        help="Degradation levels to generate (default: L1 L2 L3)")
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print()

    for level_name in args.levels:
        if level_name not in DEGRADATION_LEVELS:
            print(f"Unknown level: {level_name} (choices: {list(DEGRADATION_LEVELS.keys())})")
            continue
        generate_degraded(args.source, args.output, level_name)

    print(f"\nDone. Degraded data at: {args.output}/")


if __name__ == "__main__":
    main()
