from itertools import combinations

import numpy as np
import pandas as pd

from common import N_BOOT, OUT_DIR, RATERS, SEED, cluster_indices, load_evaluation_set, resample


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    p_keep = (np.mean(a == "KEEP") + np.mean(b == "KEEP")) / 2
    pe = 2 * p_keep * (1 - p_keep)
    return float("nan") if 1 - pe < 1e-12 else float((po - pe) / (1 - pe))


def panel_ac1(labels: dict[str, np.ndarray]) -> float:
    return float(np.nanmean([gwet_ac1(labels[a], labels[b]) for a, b in combinations(labels, 2)]))


def delta_with_ci(human: dict, panel: dict, groups: np.ndarray, rng) -> tuple[float, float, float, float]:
    mixed = panel_ac1(panel)
    uniq, idx_by_group = cluster_indices(groups)
    deltas = []
    for _ in range(N_BOOT):
        idx = resample(rng, uniq, idx_by_group)
        deltas.append(panel_ac1({k: v[idx] for k, v in panel.items()})
                      - panel_ac1({k: v[idx] for k, v in human.items()}))
    lo, hi = np.nanpercentile(deltas, [5, 95])
    return mixed, mixed - panel_ac1(human), float(lo), float(hi)


def significance(lo: float, hi: float) -> str:
    return "degradation" if hi < 0 else ("improvement" if lo > 0 else "ns")


def main():
    df, outputs = load_evaluation_set()
    groups = df["query"].values
    human = {r: df[f"meta_{r}"].values for r in RATERS}
    reference = panel_ac1(human)
    print(f"n = {len(df)}, reference panel AC1 = {reference:.4f}")

    pairwise = pd.DataFrame(
        [{"pair": f"{a} - {b}", "ac1": round(gwet_ac1(human[a], human[b]), 3)} for a, b in combinations(RATERS, 2)]
    ).sort_values("ac1", ascending=False)

    loo, aori = [], []
    for name, out in outputs.items():
        llm = out["decision"].values
        for replaced in RATERS:
            rng = np.random.default_rng(SEED)
            panel = {r: (llm if r == replaced else human[r]) for r in RATERS}
            mixed, delta, lo, hi = delta_with_ci(human, panel, groups, rng)
            loo.append({"model": name, "replaced_rater": replaced, "panel_ac1": round(mixed, 4),
                        "delta": round(delta, 4), "ci_5": round(lo, 4), "ci_95": round(hi, 4),
                        "significance": significance(lo, hi)})

        rng = np.random.default_rng(SEED)
        mixed, delta, lo, hi = delta_with_ci(human, {**human, "LLM": llm}, groups, rng)
        aori.append({"model": name, "panel_ac1_5_raters": round(mixed, 4), "delta": round(delta, 4),
                     "ci_5": round(lo, 4), "ci_95": round(hi, 4), "significance": significance(lo, hi)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for title, table, filename in [("Pairwise human AC1", pairwise, "pairwise_human_ac1.csv"),
                                   ("Leave-one-rater-out", pd.DataFrame(loo), "leave_one_rater_out.csv"),
                                   ("Add-one-rater-in", pd.DataFrame(aori), "add_one_rater_in.csv")]:
        table.to_csv(OUT_DIR / filename, index=False)
        print(f"\n{title}\n{table.to_string(index=False)}")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
