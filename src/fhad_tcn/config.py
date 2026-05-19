from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_feature_cols(cfg: dict) -> list[str]:
    f = cfg["features"]
    return f["au_regression"] + f["au_binary"] + f["pose"]
