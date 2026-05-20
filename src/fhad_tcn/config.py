from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    config_dir = path.parent
    for key in ("train_dir", "dev_dir", "checkpoints_dir", "cleaned_data_root"):
        if key in cfg["data"]:
            cfg["data"][key] = str((config_dir / cfg["data"][key]).resolve())
    return cfg


def get_feature_cols(cfg: dict) -> list[str]:
    f = cfg["features"]
    return f["au_regression"] + f["au_binary"] + f["pose"]
