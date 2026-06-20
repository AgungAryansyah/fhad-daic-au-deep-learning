def get_modality(cfg: dict) -> str:
    return cfg.get("features", {}).get("modality", "au")


def get_feature_cols(cfg: dict) -> list[str]:
    f = cfg["features"]
    if get_modality(cfg) == "egemaps":
        return f["egemaps"]
    return f["au_regression"] + f["au_binary"] + f["pose"]
