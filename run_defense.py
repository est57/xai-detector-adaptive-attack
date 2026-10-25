import json
import time

import numpy as np
import pandas as pd
import torch

import config
from src.attacks import adaptive_attack, make_success_fn, run_attack
from src.constraints import DomainConstraints
from src.data import load_dataset
from src.detector import ExplanationDetector
from src.experiment import attack_suite, evaluate_attack, freeze
from src.metrics import calibrate_threshold
from src.models import MLP, predict
from src.plots import plot_defense, plot_rounds


def train_detector(ids, n_features, x_clean, x_adv, x_val):
    """Melatih detektor baru dan mengkalibrasi threshold pada data validasi."""
    detector = ExplanationDetector(ids, n_features, config.DET_HIDDEN, config.IG_STEPS)
    detector.fit(x_clean, x_adv, config.DET_EPOCHS, config.DET_LR, config.BATCH_SIZE, config.SEED)
    freeze(detector)
    threshold = calibrate_threshold(detector.score(x_val), config.DET_TARGET_FPR)
    return detector, threshold


def probe(ids, detector, threshold, constraints, x_atk, x_eval, r):
    """Serangan adaptif singkat pada data evaluasi untuk memantau kekuatan detektor per ronde."""
    success = make_success_fn(ids, detector, threshold)
    x_adv, _, _ = adaptive_attack(ids, detector, x_atk, constraints, config.PROBE_EPS, config.ATTACK_STEPS,
                                  success, config.PROBE_LAMBDAS)
    row = evaluate_attack(ids, detector, threshold, constraints, x_atk, x_adv, config.PROBE_EPS, "adaptive")
    row.update({"round": r, "threshold": threshold,
                "fpr_clean_eval": float((detector.score(x_eval) >= threshold).mean())})
    print(f"  ronde {r}: full_evasion={row['full_evasion_rate']:.4f}  ids_evasion={row['ids_evasion_rate']:.4f}  "
          f"threshold={threshold:.4f}  FPR={row['fpr_clean_eval']:.4f}")
    return row


def main():
    config.setup()
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED + 1)
    torch.set_num_threads(config.THREADS)
    start = time.time()

    # Langkah 1: muat dataset, model IDS, detektor awal, dan indeks dari run_experiment.py
    print("[1] Memuat dataset dan model dari eksperimen utama")
    data = load_dataset(config.DATASET)
    constraints = DomainConstraints(data)
    n_features = data.x_train.shape[1]
    idx = np.load(config.RESULTS_DIR / "indices.npz")
    x_val, x_eval = data.x_test[idx["val_idx"]], data.x_test[idx["eval_idx"]]
    x_atk = torch.as_tensor(x_eval[idx["atk_idx"]])

    ids = MLP(n_features, config.IDS_HIDDEN)
    ids.load_state_dict(torch.load(config.RESULTS_DIR / "ids.pt"))
    freeze(ids)
    detector = ExplanationDetector(ids, n_features, config.DET_HIDDEN, config.IG_STEPS)
    detector.load_state_dict(torch.load(config.RESULTS_DIR / "detector.pt"))
    freeze(detector)
    threshold = float(idx["threshold"])
    transfer_samples = torch.load(config.RESULTS_DIR / "adaptive_samples.pt")

    # Langkah 2: siapkan data latih bersih dan sampel non-adaptif untuk detektor
    print("[2] Menyiapkan data latih detektor")
    train_prob = predict(ids, data.x_train)
    mal_idx = np.where((data.y_train == 1) & (train_prob >= 0.5))[0]
    mal_idx = rng.choice(mal_idx, config.DET_TRAIN_SAMPLES, replace=False)
    clean_idx = rng.choice(len(data.x_train), config.DET_TRAIN_SAMPLES, replace=False)
    chunks = np.array_split(mal_idx, len(config.DET_TRAIN_EPS))
    ids_only = make_success_fn(ids)
    adv_pool = [torch.cat([
        run_attack(ids, data.x_train[c], constraints, eps, config.ATTACK_STEPS, ids_only)[0]
        for c, eps in zip(chunks, config.DET_TRAIN_EPS)
    ])]

    # Langkah 3: pelatihan ulang detektor dengan sampel adaptif secara iteratif
    print(f"[3] Melatih ulang detektor dengan sampel adaptif (probe eps={config.PROBE_EPS})")
    round_rows = [probe(ids, detector, threshold, constraints, x_atk, x_eval, 0)]
    for r in range(1, config.DEFENSE_ROUNDS + 1):
        t0 = time.time()
        success = make_success_fn(ids, detector, threshold)
        x_adaptive = torch.cat([
            run_attack(ids, data.x_train[c], constraints, eps, config.ATTACK_STEPS, success,
                       detector, config.DEFENSE_LAMBDA)[0]
            for c, eps in zip(chunks, config.DET_TRAIN_EPS)
        ])
        evaded = float((detector.score(x_adaptive) < threshold).mean())
        adv_pool.append(x_adaptive)
        detector, threshold = train_detector(ids, n_features, data.x_train[clean_idx],
                                             torch.cat(adv_pool), x_val)
        row = probe(ids, detector, threshold, constraints, x_atk, x_eval, r)
        row["train_adaptive_evaded_prev"] = evaded
        round_rows.append(row)
        fpr = row["fpr_clean_eval"]
        print(f"           sampel latih adaptif lolos detektor sebelumnya={evaded:.4f}  ({time.time() - t0:.1f}s)")

    # Langkah 4: uji transfer, sampel adaptif terhadap detektor awal diuji ke detektor baru
    print("[4] Uji transfer sampel adaptif lama ke detektor baru")
    transfer_rows = []
    for eps, x_adv in transfer_samples.items():
        row = evaluate_attack(ids, detector, threshold, constraints, x_atk, x_adv, eps, "transfer")
        transfer_rows.append(row)
        print(f"  eps={eps:<5} transfer      full_evasion={row['full_evasion_rate']:.4f}")

    # Langkah 5: serangan adaptif baru terhadap detektor yang sudah dilatih ulang
    print("[5] Serangan adaptif terhadap detektor yang dilatih ulang")
    rows, lambda_rows, _ = attack_suite(ids, detector, threshold, constraints, x_atk, config.ATTACK_EPS_LIST,
                                        config.ATTACK_STEPS, config.ADAPTIVE_LAMBDAS)

    # Langkah 6: simpan hasil dan grafik perbandingan
    defense = pd.DataFrame(rows + transfer_rows)
    defense.to_csv(config.RESULTS_DIR / "defense_results.csv", index=False)
    pd.DataFrame(lambda_rows).to_csv(config.RESULTS_DIR / "defense_lambda_results.csv", index=False)
    rounds = pd.DataFrame(round_rows)
    rounds.to_csv(config.RESULTS_DIR / "defense_rounds.csv", index=False)
    plot_rounds(rounds, config.RESULTS_DIR / "defense_rounds.png")
    summary = {"rounds": config.DEFENSE_ROUNDS, "defense_lambda": config.DEFENSE_LAMBDA,
               "detector_threshold": threshold, "detector_fpr_clean_eval": fpr}
    (config.RESULTS_DIR / "defense_summary.json").write_text(json.dumps(summary, indent=2))
    baseline = pd.read_csv(config.RESULTS_DIR / "attack_results.csv")
    plot_defense(baseline, defense, config.RESULTS_DIR / "defense_results.png")
    print(f"[6] Selesai dalam {time.time() - start:.1f}s. Hasil disimpan di {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
