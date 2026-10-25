import time

import numpy as np
import pandas as pd
import torch

import config
from src.attacks import AttributionTarget, make_success_fn, run_attack, stealth_attack
from src.constraints import DomainConstraints
from src.data import load_dataset
from src.detector import ExplanationDetector
from src.experiment import evaluate_attack, freeze
from src.explain import explain_batches
from src.models import MLP, predict


def mean_distance(target, x_adv, ids, batch_size=1024):
    """Median jarak atribusi ke pusat benign, hanya pada sampel yang lolos IDS."""
    evade = predict(ids, x_adv) < 0.5
    if not evade.any():
        return float("nan")
    x = x_adv[torch.as_tensor(evade)]
    dist = torch.cat([target.distance(x[i:i + batch_size]).detach() for i in range(0, len(x), batch_size)])
    return float(dist.median())


def main():
    config.setup()
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED + 2)
    torch.set_num_threads(config.THREADS)
    start = time.time()

    # Langkah 1: muat dataset, model IDS, detektor awal, dan indeks dari run_experiment.py
    print(f"[1] Memuat {config.DATASET} dan model dari eksperimen utama")
    data = load_dataset(config.DATASET)
    constraints = DomainConstraints(data)
    n_features = data.x_train.shape[1]
    idx = np.load(config.RESULTS_DIR / "indices.npz")
    x_atk = torch.as_tensor(data.x_test[idx["eval_idx"]][idx["atk_idx"]])
    ids = freeze(MLP(n_features, config.IDS_HIDDEN))
    ids.load_state_dict(torch.load(config.RESULTS_DIR / "ids.pt"))
    detector = ExplanationDetector(ids, n_features, config.DET_HIDDEN, config.IG_STEPS)
    detector.load_state_dict(torch.load(config.RESULTS_DIR / "detector.pt"))
    freeze(detector)
    threshold = float(idx["threshold"])
    adaptive_samples = torch.load(config.RESULTS_DIR / "adaptive_samples.pt")

    # Langkah 2: pusat atribusi trafik benign (pengetahuan penyerang, tanpa akses ke detektor)
    print("[2] Menghitung pusat atribusi benign")
    benign_idx = rng.choice(np.where(data.y_train == 0)[0], config.DET_TRAIN_SAMPLES, replace=False)
    target = AttributionTarget(ids, explain_batches(ids, data.x_train[benign_idx], config.IG_STEPS), config.IG_STEPS)

    # Langkah 3: serangan bergaya ASA-IDS, dibandingkan dengan non-adaptif dan adaptif pada detektor awal
    print("[3] Menjalankan serangan bergaya ASA-IDS")
    ids_only = make_success_fn(ids)
    rows, mu_rows = [], []
    for eps in config.ATTACK_EPS_LIST:
        t0 = time.time()
        x_stealth, _, per_mu = stealth_attack(ids, x_atk, constraints, eps, config.ATTACK_STEPS, target,
                                              config.STEALTH_MUS)
        x_plain, _ = run_attack(ids, x_atk, constraints, eps, config.ATTACK_STEPS, ids_only)
        for name, x_adv in (("attribution_stealth", x_stealth), ("non_adaptive", x_plain),
                            ("adaptive", adaptive_samples[eps])):
            row = evaluate_attack(ids, detector, threshold, constraints, x_atk, x_adv, eps, name)
            row["attr_distance"] = mean_distance(target, x_adv, ids)
            rows.append(row)
            print(f"  eps={eps:<5} {name:<20} ids_evasion={row['ids_evasion_rate']:.4f}  "
                  f"detected={row['detector_detection_rate']:.4f}  full_evasion={row['full_evasion_rate']:.4f}  "
                  f"attr_dist={row['attr_distance']:.3f}  violations={row['violations']}")
        mu_rows += [{"eps": eps, "mu": mu, "ids_evasion_rate": r} for mu, r in per_mu.items()]
        print(f"  ({time.time() - t0:.1f}s)")

    # Langkah 4: simpan hasil
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "baseline_results.csv", index=False)
    pd.DataFrame(mu_rows).to_csv(config.RESULTS_DIR / "baseline_mu_results.csv", index=False)
    print(f"[4] Selesai dalam {time.time() - start:.1f}s. Hasil disimpan di {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
