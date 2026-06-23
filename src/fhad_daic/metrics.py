CLINICAL_VISUAL_FEATURES = {
    "AU06_r", "AU12_r", "AU14_r",
    "AU04_r", "AU07_r", "AU01_r",
    "pose_Ry", "pose_Rz",
}

CLINICAL_AUDIO_FEATURES = {
    "F0semitoneFrom27.5Hz_sma3nz",
    "Loudness_sma3",
    "HNRdBACF_sma3nz",
    "shimmerLocaldB_sma3nz",
    "mfcc1_sma3",
    "jitterLocal_sma3nz",
}


def compute_cas(feature_names: list[str], importance: dict[str, float], top_k: int, clinical_features: set[str] | None = None) -> float:
    if clinical_features is None:
        clinical_features = CLINICAL_VISUAL_FEATURES
    sorted_features = sorted(importance, key=importance.get, reverse=True)[:top_k]
    hits = len(set(sorted_features) & set(clinical_features))
    return hits / top_k


def compute_cas_at_k(feature_names: list[str], importance: dict[str, float], ks: list[int], clinical_features: set[str] | None = None) -> dict[int, float]:
    return {k: compute_cas(feature_names, importance, k, clinical_features) for k in ks}


def compute_pgi_pgu(feature_names: list[str], importance: dict[str, float], top_k: int, n_random: int = 3) -> tuple[float, float]:
    sorted_features = sorted(importance, key=importance.get, reverse=True)
    top_set = set(sorted_features[:top_k])
    all_features = set(feature_names)
    unimportant = list(all_features - top_set - set(["confidence", "confidence_mean", "confidence_std", "confidence_min"]))
    pgi = len(top_set) / len(all_features) if all_features else 0.0
    pgu = len(set(unimportant[:top_k])) / len(all_features) if all_features else 0.0
    return pgi, pgu
