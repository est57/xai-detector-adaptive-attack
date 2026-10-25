import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

METRICS = ["ids_evasion_rate", "detector_detection_rate", "full_evasion_rate"]
REQUIRED = ["attack_results.csv", "summary.json", "defense_results.csv", "defense_rounds.csv"]


def seed_dirs():
    """Folder seed yang sudah lengkap (eksperimen utama dan pertahanan selesai)."""
    dirs = sorted(config.DATASET_DIR.glob("seed_*"))
    return [d for d in dirs if all((d / f).exists() for f in REQUIRED)]


def load_all(dirs, filename):
    """Menggabungkan file CSV yang sama dari semua seed."""
    return pd.concat([pd.read_csv(d / filename).assign(seed=d.name) for d in dirs], ignore_index=True)


def mean_std(df, keys, cols):
    """Rata-rata dan simpangan baku (ddof=1) per kelompok."""
    out = df.groupby(keys)[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index()


def fmt(df, keys, cols):
    """Tabel ringkas dalam format persen: rata-rata +- simpangan baku."""
    table = df[keys].copy()
    for c in cols:
        table[c] = [f"{m * 100:5.1f} +- {s * 100:4.1f}" for m, s in zip(df[f"{c}_mean"], df[f"{c}_std"])]
    return table.to_string(index=False)


def plot_errorbars(main, defense, rounds, path):
    """Grafik rata-rata dengan error bar simpangan baku."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    series = [(main, "non_adaptive", "o--", "non-adaptive vs initial detector"),
              (main, "adaptive", "s-", "adaptive vs initial detector"),
              (defense, "transfer", "^:", "transfer to retrained detector"),
              (defense, "adaptive", "D-", "adaptive vs retrained detector")]
    for df, name, style, label in series:
        sub = df[df["attack"] == name].sort_values("eps")
        axes[0].errorbar(sub["eps"], sub["full_evasion_rate_mean"], yerr=sub["full_evasion_rate_std"],
                         fmt=style, capsize=3, label=label)
    axes[0].set_title("Full evasion (IDS + detector)")
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("fraction of samples")
    axes[0].legend(fontsize=8)

    for col, style, label in (("full_evasion_rate", "s-", "full evasion (adaptive)"),
                              ("detector_detection_rate", "o--", "flagged by detector"),
                              ("fpr_clean_eval", "^:", "FPR on clean data")):
        axes[1].errorbar(rounds["round"], rounds[f"{col}_mean"], yerr=rounds[f"{col}_std"],
                         fmt=style, capsize=3, label=label)
    axes[1].set_title(f"Detector retraining (eps={config.PROBE_EPS})")
    axes[1].set_xlabel("round (0 = initial detector)")
    axes[1].set_xticks(rounds["round"])
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    # Langkah 1: kumpulkan hasil dari setiap seed
    config.setup(make_dirs=False)
    dirs = seed_dirs()
    out_dir = config.DATASET_DIR / "aggregate"
    out_dir.mkdir(exist_ok=True)
    print(f"Dataset: {config.DATASET}  Seed: {', '.join(d.name for d in dirs)} (n={len(dirs)})")

    # Langkah 2: performa IDS dan FPR detektor awal
    summaries = []
    for d in dirs:
        s = json.loads((d / "summary.json").read_text())
        summaries.append({**s["ids_clean_eval"], "detector_fpr": s["detector_fpr_clean_eval"]})
    ids = pd.DataFrame(summaries).agg(["mean", "std"]).T
    ids.to_csv(out_dir / "ids_mean_std.csv")
    print("\nIDS pada data evaluasi bersih dan FPR detektor awal")
    print(ids.to_string(float_format=lambda v: f"{v:.4f}"))

    # Langkah 3: eksperimen utama, pertahanan, dan tren per ronde
    main_df = mean_std(load_all(dirs, "attack_results.csv"), ["attack", "eps"], METRICS)
    defense_df = mean_std(load_all(dirs, "defense_results.csv"), ["attack", "eps"], METRICS)
    rounds_df = mean_std(load_all(dirs, "defense_rounds.csv"), ["round"],
                         METRICS + ["fpr_clean_eval", "threshold"])
    main_df.to_csv(out_dir / "main_mean_std.csv", index=False)
    defense_df.to_csv(out_dir / "defense_mean_std.csv", index=False)
    rounds_df.to_csv(out_dir / "rounds_mean_std.csv", index=False)

    print("\nEksperimen utama (detektor awal), persen")
    print(fmt(main_df, ["attack", "eps"], METRICS))
    print("\nSetelah pelatihan ulang detektor, persen")
    print(fmt(defense_df, ["attack", "eps"], METRICS))
    print(f"\nTren per ronde (serangan adaptif eps={config.PROBE_EPS}), persen")
    print(fmt(rounds_df, ["round"], ["full_evasion_rate", "detector_detection_rate", "fpr_clean_eval"]))

    # Langkah 4: grafik dengan error bar
    plot_errorbars(main_df, defense_df, rounds_df, out_dir / "aggregate.png")

    # Langkah 5: pembanding serangan bergaya ASA-IDS (jika run_baseline.py sudah dijalankan di semua seed)
    if all((d / "baseline_results.csv").exists() for d in dirs):
        base = load_all(dirs, "baseline_results.csv")
        base_df = mean_std(base, ["attack", "eps"], METRICS)
        dist_df = base.groupby(["attack", "eps"])["attr_distance"].agg(["mean", "std"]).reset_index()
        base_df = base_df.merge(dist_df.rename(columns={"mean": "attr_distance_mean", "std": "attr_distance_std"}),
                                on=["attack", "eps"])
        base_df.to_csv(out_dir / "baseline_mean_std.csv", index=False)
        print("\nPembanding serangan bergaya ASA-IDS terhadap detektor awal, persen")
        print(fmt(base_df, ["attack", "eps"], METRICS))
        print("\nMedian jarak atribusi ke pusat benign (sampel yang lolos IDS)")
        print(dist_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nHasil agregasi disimpan di {out_dir}")


if __name__ == "__main__":
    main()
