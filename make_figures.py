"""Membuat ulang seluruh grafik dari file hasil uji coba (CSV) tanpa menjalankan ulang serangan."""
import pandas as pd

import config
from src.plots import plot_defense, plot_paper_attacks, plot_paper_rounds, plot_results, plot_rounds

DATASETS = ("nsl_kdd", "unsw_nb15")


def main():
    # Langkah 1: grafik per seed dari attack_results.csv, defense_rounds.csv, dan defense_results.csv
    for name in DATASETS:
        for d in sorted((config.RESULTS_ROOT / name).glob("seed_*")):
            attack = pd.read_csv(d / "attack_results.csv")
            plot_results(attack, d / "attack_results.png")
            plot_rounds(pd.read_csv(d / "defense_rounds.csv"), d / "defense_rounds.png")
            plot_defense(attack, pd.read_csv(d / "defense_results.csv"), d / "defense_results.png")
            print(f"grafik per seed: {d}")

    # Langkah 2: grafik artikel dari hasil agregasi lima seed (aggregate.py)
    out = config.RESULTS_ROOT / "figures"
    out.mkdir(exist_ok=True)
    agg = {name: config.RESULTS_ROOT / name / "aggregate" for name in DATASETS}
    plot_paper_attacks({n: pd.read_csv(p / "baseline_mean_std.csv") for n, p in agg.items()}, out / "fig_attacks.png")
    plot_paper_rounds({n: pd.read_csv(p / "rounds_mean_std.csv") for n, p in agg.items()}, out / "fig_retraining.png")
    print(f"grafik artikel: {out}")


if __name__ == "__main__":
    main()
