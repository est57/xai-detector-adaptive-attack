import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    ("ids_evasion_rate", "IDS evasion"),
    ("detector_detection_rate", "Flagged by detector"),
    ("full_evasion_rate", "Full evasion (IDS + detector)"),
]

DATASET_NAMES = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15"}

# Gaya grafik untuk artikel: Times, lebar satu kolom IEEE (3,5 inci)
PAPER_STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
}


def plot_results(results, path):
    """Grafik perbandingan serangan non-adaptif dan adaptif terhadap epsilon."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4), sharey=True)
    for ax, (col, title) in zip(axes, METRICS):
        for name, style, label in (("non_adaptive", "o--", "non-adaptive"), ("adaptive", "s-", "adaptive")):
            subset = results[results["attack"] == name]
            ax.plot(subset["eps"], subset[col], style, label=label)
        ax.set_title(title)
        ax.set_xlabel("epsilon")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of samples")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rounds(rounds, path):
    """Grafik tren serangan adaptif dan FPR detektor terhadap ronde pelatihan ulang."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds["round"], rounds["full_evasion_rate"], "s-", label="full evasion (adaptive)")
    ax.plot(rounds["round"], rounds["detector_detection_rate"], "o--", label="flagged by detector")
    ax.plot(rounds["round"], rounds["fpr_clean_eval"], "^:", label="FPR on clean data")
    ax.set_xlabel("retraining round (0 = initial detector)")
    ax.set_ylabel("fraction of samples")
    ax.set_xticks(rounds["round"])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_defense(baseline, defense, path):
    """Grafik perbandingan sebelum dan sesudah detektor dilatih ulang dengan sampel adaptif."""
    series = [
        (baseline, "adaptive", "s-", "adaptive vs initial detector"),
        (defense, "transfer", "^:", "transfer to retrained detector"),
        (defense, "adaptive", "D-", "adaptive vs retrained detector"),
        (defense, "non_adaptive", "o--", "non-adaptive vs retrained detector"),
    ]
    panels = [("full_evasion_rate", "Full evasion (IDS + detector)"),
              ("detector_detection_rate", "Flagged by detector")]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        for df, name, style, label in series:
            subset = df[df["attack"] == name].sort_values("eps")
            ax.plot(subset["eps"], subset[col], style, label=label)
        ax.set_title(title)
        ax.set_xlabel("epsilon")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of samples")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _paper_panels(frames, path, draw, xlabel, ylabel, ylim, ncol):
    """Kerangka grafik artikel: satu panel per dataset, legenda bersama di atas."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(frames), figsize=(3.5, 1.75), sharey=True)
        for ax, (key, df) in zip(axes, frames.items()):
            draw(ax, df)
            ax.set_title(DATASET_NAMES.get(key, key), fontsize=7)
            ax.set_xlabel(xlabel)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, linewidth=0.4)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
        fig.savefig(path, dpi=600)
        plt.close(fig)


def plot_paper_attacks(baselines, path):
    """Full evasion rate (rata-rata +- simpangan baku antar seed) tiga serangan terhadap epsilon."""
    attacks = (("non_adaptive", "Non-adaptive", "o", "--", "#7f7f7f"),
               ("attribution_stealth", "ASA-style", "^", ":", "#1f77b4"),
               ("adaptive", "Adaptive (ours)", "s", "-", "#d62728"))

    def draw(ax, df):
        for name, label, marker, ls, color in attacks:
            d = df[df["attack"] == name].sort_values("eps")
            ax.errorbar(d["eps"], d["full_evasion_rate_mean"] * 100, yerr=d["full_evasion_rate_std"] * 100,
                        marker=marker, linestyle=ls, color=color, capsize=1.5, elinewidth=0.6, label=label)
        ax.set_xticks([0.02, 0.1, 0.2, 0.3])

    _paper_panels(baselines, path, draw, "Perturbation budget ε", "Full evasion rate (%)", (0, 90), 3)


def plot_paper_rounds(rounds, path):
    """Full evasion rate dan detection rate serangan adaptif pada setiap ronde pelatihan ulang."""
    series = (("full_evasion_rate", "Full evasion rate", "s", "-", "#d62728"),
              ("detector_detection_rate", "Detection rate", "o", "--", "#1f77b4"))

    def draw(ax, df):
        for col, label, marker, ls, color in series:
            ax.errorbar(df["round"], df[f"{col}_mean"] * 100, yerr=df[f"{col}_std"] * 100,
                        marker=marker, linestyle=ls, color=color, capsize=1.5, elinewidth=0.6, label=label)
        ax.set_xticks(df["round"])

    _paper_panels(rounds, path, draw, "Retraining round", "Rate (%)", (0, 100), 2)
