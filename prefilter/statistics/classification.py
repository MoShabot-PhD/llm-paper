from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from baselines import classify_variant_a, classify_variant_b
from common import N_BOOT, N_SPLITS, OUT_DIR, SEED, cluster_indices, load_evaluation_set, metrics, resample

THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
FIXED_THRESHOLD = 0.5
BASELINES = ["Regex A (stoplist)", "Regex B (stoplist+keyword)"]


def pick_threshold(y: np.ndarray, keep_score: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = metrics(y, (keep_score >= t).astype(int))[2]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def fold_cv_predict(keep_score: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    preds = np.zeros(len(y), dtype=int)
    for train, test in GroupKFold(n_splits=N_SPLITS).split(keep_score, y, groups):
        t = pick_threshold(y[train], keep_score[train])
        preds[test] = (keep_score[test] >= t).astype(int)
    return preds


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    uniq, idx_by_group = cluster_indices(groups)
    samples = []
    for _ in range(N_BOOT):
        idx = resample(rng, uniq, idx_by_group)
        if y[idx].sum() == 0:
            continue
        samples.append(metrics(y[idx], pred[idx]))
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def paired_difference(pred_a, pred_b, y, groups, metric_idx: int, rng) -> tuple[float, float, float]:
    uniq, idx_by_group = cluster_indices(groups)
    diffs = []
    for _ in range(N_BOOT):
        idx = resample(rng, uniq, idx_by_group)
        if y[idx].sum() in (0, len(idx)):
            continue
        diffs.append(metrics(y[idx], pred_a[idx])[metric_idx] - metrics(y[idx], pred_b[idx])[metric_idx])
    point = metrics(y, pred_a)[metric_idx] - metrics(y, pred_b)[metric_idx]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def threshold_free(y: np.ndarray, score: np.ndarray, groups: np.ndarray, rng) -> dict:
    uniq, idx_by_group = cluster_indices(groups)
    samples = []
    for _ in range(N_BOOT):
        idx = resample(rng, uniq, idx_by_group)
        if y[idx].sum() in (0, len(idx)):
            continue
        samples.append((roc_auc_score(y[idx], score[idx]), average_precision_score(y[idx], score[idx])))
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {"roc_auc": round(roc_auc_score(y, score), 3), "roc_auc_ci": f"[{lo[0]:.3f}, {hi[0]:.3f}]",
            "pr_auc": round(average_precision_score(y, score), 3), "pr_auc_ci": f"[{lo[1]:.3f}, {hi[1]:.3f}]"}


def main():
    df, outputs = load_evaluation_set()
    y = df["y"].values
    groups = df["query"].values
    text = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)

    preds = {name: fold_cv_predict(out["keep_score"].values.astype(float), y, groups)
             for name, out in outputs.items()}
    preds[BASELINES[0]] = (text.apply(classify_variant_a) == "KEEP").astype(int).values
    preds[BASELINES[1]] = (text.apply(classify_variant_b) == "KEEP").astype(int).values
    print(f"n = {len(y)}, KEEP = {int(y.sum())} ({y.mean():.1%})")

    rng = np.random.default_rng(SEED)
    performance = []
    for name, pred in preds.items():
        (p, r, f1), (lo, hi) = metrics(y, pred), bootstrap_ci(y, pred, groups, rng)
        performance.append({"method": name, "precision": round(p, 3),
                            "precision_ci": f"[{lo[0]:.3f}, {hi[0]:.3f}]", "recall": round(r, 3),
                            "recall_ci": f"[{lo[1]:.3f}, {hi[1]:.3f}]", "f1": round(f1, 3),
                            "f1_ci": f"[{lo[2]:.3f}, {hi[2]:.3f}]"})

    llms = list(outputs)
    pairs = [(a, b) for a in llms for b in BASELINES] + list(combinations(llms, 2))
    comparisons = []
    for metric, metric_idx in [("F1", 2), ("Recall", 1)]:
        for a, b in pairs:
            diff, lo, hi = paired_difference(preds[a], preds[b], y, groups, metric_idx, rng)
            comparisons.append({"comparison": f"{a} - {b}", "metric": metric, "delta": round(diff, 3),
                                "ci": f"[{lo:+.3f}, {hi:+.3f}]", "significant": "yes" if lo > 0 or hi < 0 else "no"})

    auc = pd.DataFrame([{"method": name, **threshold_free(y, out["keep_score"].values.astype(float), groups, rng)}
                        for name, out in outputs.items()])
    print(f"\nPR-AUC of a random classifier = prevalence = {y.mean():.3f}")

    threshold_rows = []
    for name, out in outputs.items():
        fixed = (out["keep_score"].values.astype(float) >= FIXED_THRESHOLD).astype(int)
        for metric, metric_idx in [("Precision", 0), ("Recall", 1), ("F1", 2)]:
            diff, lo, hi = paired_difference(preds[name], fixed, y, groups, metric_idx, rng)
            threshold_rows.append({"model": name, "metric": metric,
                                   "fixed_0.5": round(metrics(y, fixed)[metric_idx], 3),
                                   "tuned": round(metrics(y, preds[name])[metric_idx], 3),
                                   "delta_tuned_minus_fixed": round(diff, 3), "ci": f"[{lo:+.3f}, {hi:+.3f}]",
                                   "significant": "yes" if lo > 0 or hi < 0 else "no"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"url": df["url"], "query": groups, "y": y, **preds}).to_csv(OUT_DIR / "predictions.csv", index=False)
    for title, table, filename in [("Classification performance", pd.DataFrame(performance), "classification_performance.csv"),
                                   ("Paired cluster-bootstrap comparisons", pd.DataFrame(comparisons), "pairwise_comparisons.csv"),
                                   ("Threshold-free performance", auc, "threshold_free_performance.csv"),
                                   ("Tuned vs fixed threshold", pd.DataFrame(threshold_rows), "threshold_comparison.csv")]:
        table.to_csv(OUT_DIR / filename, index=False)
        print(f"\n{title}\n{table.to_string(index=False)}")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
