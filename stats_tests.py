"""Uji t berpasangan antar-seed untuk perbandingan utama pada eps = 0,3, dengan koreksi Holm."""
import numpy as np
import pandas as pd
from scipy import stats

import config

DATASETS = ("nsl_kdd", "unsw_nb15")
EPS = 0.3

# (nama perbandingan, (file, serangan) kelompok A, (file, serangan) kelompok B, metrik)
COMPARISONS = [
    ("adaptive_vs_non_adaptive", ("baseline_results.csv", "adaptive"), ("baseline_results.csv", "non_adaptive")),
    ("adaptive_vs_asa_style", ("baseline_results.csv", "adaptive"), ("baseline_results.csv", "attribution_stealth")),
    ("regenerated_vs_transfer", ("defense_results.csv", "adaptive"), ("defense_results.csv", "transfer")),
    ("regenerated_vs_initial_adaptive", ("defense_results.csv", "adaptive"), ("baseline_results.csv", "adaptive")),
    ("non_adaptive_retrained_vs_initial", ("defense_results.csv", "non_adaptive"), ("baseline_results.csv", "non_adaptive")),
]


def seed_values(name, filename, attack):
    """Full evasion rate pada eps = 0,3 untuk setiap seed, diurutkan menurut seed."""
    values = {}
    for d in sorted((config.RESULTS_ROOT / name).glob("seed_*")):
        df = pd.read_csv(d / filename)
        row = df[(df["attack"] == attack) & (np.isclose(df["eps"], EPS))]
        values[d.name] = float(row["full_evasion_rate"].iloc[0])
    return values


def holm(pvalues):
    """Koreksi Holm-Bonferroni untuk sekumpulan nilai p."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main():
    # Langkah 1: kumpulkan pasangan nilai per seed dan hitung uji t berpasangan
    rows = []
    for name in DATASETS:
        for label, (file_a, attack_a), (file_b, attack_b) in COMPARISONS:
            a, b = seed_values(name, file_a, attack_a), seed_values(name, file_b, attack_b)
            seeds = sorted(set(a) & set(b))
            diff = np.array([a[s] - b[s] for s in seeds]) * 100
            t, p = stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            rows.append({
                "dataset": name, "comparison": label, "n_seeds": len(seeds),
                "mean_a": np.mean([a[s] for s in seeds]) * 100, "mean_b": np.mean([b[s] for s in seeds]) * 100,
                "mean_diff": diff.mean(), "ci_low": diff.mean() - half, "ci_high": diff.mean() + half,
                "t": t, "p": p, "cohen_dz": diff.mean() / diff.std(ddof=1),
                "same_sign": int(max((diff > 0).sum(), (diff < 0).sum())),
            })

    # Langkah 2: koreksi Holm atas seluruh perbandingan, lalu simpan hasil
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    out_dir = config.RESULTS_ROOT / "stats"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "paired_tests.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nHasil disimpan di {out_dir / 'paired_tests.csv'}")


if __name__ == "__main__":
    main()
