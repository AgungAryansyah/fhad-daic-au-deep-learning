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

PERTURBATION_TYPES = ["mask", "noise", "shuffle"]


def compute_cas(feature_names: list[str], importance: dict[str, float], top_k: int, clinical_features: set[str] | None = None) -> float:
    if clinical_features is None:
        clinical_features = CLINICAL_VISUAL_FEATURES
    sorted_features = sorted(importance, key=importance.get, reverse=True)[:top_k]
    hits = len(set(sorted_features) & set(clinical_features))
    return hits / top_k


def compute_cas_at_k(feature_names: list[str], importance: dict[str, float], ks: list[int], clinical_features: set[str] | None = None) -> dict[int, float]:
    return {k: compute_cas(feature_names, importance, k, clinical_features) for k in ks}


def _top_feature_indices(feature_names: list[str], importance: dict[str, float], top_k: int) -> list[int]:
    sorted_features = sorted(importance, key=importance.get, reverse=True)
    top_set = set(sorted_features[:top_k])
    return [i for i, name in enumerate(feature_names) if name in top_set]


def _random_feature_indices(feature_names: list[str], importance: dict[str, float], top_k: int) -> list[int]:
    sorted_features = sorted(importance, key=importance.get, reverse=True)
    top_set = set(sorted_features[:top_k])
    meta = {"confidence", "confidence_mean", "confidence_std", "confidence_min"}
    unimportant = [name for name in feature_names if name not in top_set and name not in meta]
    import random
    random.seed(42)
    chosen = random.sample(unimportant, min(top_k, len(unimportant)))
    return [feature_names.index(name) for name in chosen]


def apply_perturbation(
    X: list, feature_names: list[str], importance: dict[str, float],
    perturbation_type: str, top_k: int,
) -> list:
    import copy
    X_pert = [X_np.copy() for X_np in X]

    if perturbation_type == "mask":
        feat_idx = _top_feature_indices(feature_names, importance, top_k)
        for arr in X_pert:
            arr[:, feat_idx] = 0.0

    elif perturbation_type == "mask_random":
        feat_idx = _random_feature_indices(feature_names, importance, top_k)
        for arr in X_pert:
            arr[:, feat_idx] = 0.0

    elif perturbation_type == "noise":
        import numpy as np
        rng = np.random.RandomState(42)
        for arr in X_pert:
            arr[:] = arr + rng.normal(0, 0.5, arr.shape).astype(arr.dtype)

    elif perturbation_type == "shuffle":
        import random
        random.seed(42)
        n_swap = max(1, int(len(X_pert) * 0.2))
        idx = list(range(len(X_pert)))
        random.shuffle(idx[:n_swap])
        for i in range(n_swap):
            X_pert[i], X_pert[idx[i]] = X_pert[idx[i]].copy(), X_pert[i].copy()

    return X_pert


def compute_pgi_pgu_empirical(
    model, loader_fn, X_data, feature_names, importance, top_k, criterion, device,
) -> dict[str, float]:
    import torch
    import numpy as np
    from sklearn.metrics import f1_score

    def _eval_f1(X):
        loader = loader_fn(X)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_labels.extend(yb.tolist())
        return f1_score(all_labels, all_preds, average="macro", zero_division=0)

    baseline_f1 = _eval_f1(X_data)

    results = {"baseline_f1": baseline_f1}
    for ptype in PERTURBATION_TYPES:
        X_pert = apply_perturbation(X_data, feature_names, importance, ptype, top_k)
        results[f"pg_{ptype}"] = _eval_f1(X_pert)

    mask_rand = apply_perturbation(X_data, feature_names, importance, "mask_random", top_k)
    results["pg_mask_random"] = _eval_f1(mask_rand)

    pgi = max(0.0, results["pg_mask"] - results["pg_mask_random"])
    pgu = max(0.0, baseline_f1 - results["pg_mask_random"])
    results["pgi"] = pgi
    results["pgu"] = pgu
    results["pgi_pgu_ratio"] = pgi / pgu if pgu > 0 else 0.0

    for ptype in PERTURBATION_TYPES + ["mask_random"]:
        results[f"f1_drop_{ptype}"] = baseline_f1 - results[f"pg_{ptype}"]

    stability_scores = []
    for ptype in PERTURBATION_TYPES:
        drops = []
        for _ in range(3):
            X_pert = apply_perturbation(X_data, feature_names, importance, ptype, top_k)
            drops.append(baseline_f1 - _eval_f1(X_pert))
        std_d = float(np.std(drops))
        mean_d = float(np.mean(drops)) if np.mean(drops) > 0 else 1.0
        results[f"stability_{ptype}"] = 1.0 - min(std_d / mean_d, 1.0)
        stability_scores.append(results[f"stability_{ptype}"])
    results["stability"] = float(np.mean(stability_scores))

    return results
