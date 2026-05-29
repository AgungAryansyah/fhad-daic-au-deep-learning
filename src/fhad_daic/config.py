def get_feature_cols(cfg: dict) -> list[str]:
    f = cfg["features"]
    return f["au_regression"] + f["au_binary"] + f["pose"]
