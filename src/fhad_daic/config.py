def get_modality(cfg: dict) -> str:
    return cfg.get("features", {}).get("modality", "au")


def get_feature_cols(cfg: dict) -> list[str]:
    f = cfg.get("features", {})
    if get_modality(cfg) == "egemaps":
        return f.get("egemaps", [])
    cols = f.get("au_regression", []) + f.get("au_binary", []) + f.get("pose", [])
    if f.get("include_confidence", False):
        cols = cols + f.get("confidence", ["confidence"])
        if f.get("confidence_aggregates", False):
            cols = cols + ["confidence_mean", "confidence_std", "confidence_min"]
    return cols
