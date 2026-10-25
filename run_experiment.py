import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import config
from src.attacks import make_success_fn, run_attack
from src.constraints import DomainConstraints
from src.data import load_dataset
from src.detector import ExplanationDetector
from src.experiment import attack_suite, freeze
from src.metrics import calibrate_threshold, ids_metrics
from src.models import MLP, predict, train_classifier
from src.plots import plot_results


def main():
    config.setup()
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    torch.set_num_threads(config.THREADS)
    start = time.time()

    # Langkah 1: muat dataset, bagi data uji menjadi validasi (kalibrasi) dan evaluasi
    print(f"[1] Memuat dataset {config.DATASET}")
    data = load_dataset(config.DATASET)
    constraints = DomainConstraints(data)
    n_features = data.x_train.shape[1]
    val_idx, eval_idx = train_test_split(np.arange(len(data.y_test)), test_size=1 - config.VAL_FRACTION,
                                         stratify=data.y_test, random_state=config.SEED)
    x_val, x_eval, y_eval = data.x_test[val_idx], data.x_test[eval_idx], data.y_test[eval_idx]
    print(f"  train={data.x_train.shape}  val={x_val.shape}  eval={x_eval.shape}  "
          f"fitur mutable={int(constraints.mutable.sum())}/{n_features}")

    # Langkah 2: latih model IDS dan evaluasi pada data evaluasi bersih
    print("[2] Melatih model IDS")
    ids = MLP(n_features, config.IDS_HIDDEN)
    train_classifier(ids, data.x_train, data.y_train, config.IDS_EPOCHS, config.IDS_LR,
                     config.BATCH_SIZE, config.SEED)
    freeze(ids)
    ids_clean = ids_metrics(y_eval, predict(ids, x_eval))
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in ids_clean.items()))

    # Langkah 3: buat sampel adversarial non-adaptif untuk melatih detektor
    print("[3] Membuat data latih detektor")
    ids_only = make_success_fn(ids)
    train_prob = predict(ids, data.x_train)
    mal_idx = np.where((data.y_train == 1) & (train_prob >= 0.5))[0]
    mal_idx = rng.choice(mal_idx, config.DET_TRAIN_SAMPLES, replace=False)
    clean_idx = rng.choice(len(data.x_train), config.DET_TRAIN_SAMPLES, replace=False)
    chunks = np.array_split(mal_idx, len(config.DET_TRAIN_EPS))
    x_adv_train = torch.cat([
        run_attack(ids, data.x_train[idx], constraints, eps, config.ATTACK_STEPS, ids_only)[0]
        for idx, eps in zip(chunks, config.DET_TRAIN_EPS)
    ])

    # Langkah 4: latih detektor dan kalibrasi threshold pada data validasi
    print("[4] Melatih detektor berbasis penjelasan (Integrated Gradients)")
    detector = ExplanationDetector(ids, n_features, config.DET_HIDDEN, config.IG_STEPS)
    detector.fit(data.x_train[clean_idx], x_adv_train, config.DET_EPOCHS, config.DET_LR,
                 config.BATCH_SIZE, config.SEED)
    freeze(detector)
    threshold = calibrate_threshold(detector.score(x_val), config.DET_TARGET_FPR)
    eval_fpr = float((detector.score(x_eval) >= threshold).mean())
    print(f"  threshold={threshold:.4f}  FPR pada data evaluasi bersih={eval_fpr:.4f}")

    # Langkah 5: pilih sampel serangan dari data evaluasi yang terdeteksi benar oleh IDS
    eval_prob = predict(ids, x_eval)
    atk_idx = np.where((y_eval == 1) & (eval_prob >= 0.5))[0]
    atk_idx = rng.choice(atk_idx, min(config.MAX_ATTACK_SAMPLES, len(atk_idx)), replace=False)
    x_atk = torch.as_tensor(x_eval[atk_idx])
    print(f"[5] Sampel serangan uji: {len(x_atk)}")

    # Langkah 6: serangan non-adaptif dan adaptif untuk setiap epsilon
    print("[6] Menjalankan serangan")
    rows, lambda_rows, adaptive_samples = attack_suite(ids, detector, threshold, constraints, x_atk, config.ATTACK_EPS_LIST,
                                     config.ATTACK_STEPS, config.ADAPTIVE_LAMBDAS)

    # Langkah 7: simpan hasil, grafik, model, dan indeks data
    results = pd.DataFrame(rows)
    results.to_csv(config.RESULTS_DIR / "attack_results.csv", index=False)
    pd.DataFrame(lambda_rows).to_csv(config.RESULTS_DIR / "lambda_results.csv", index=False)
    summary = {"ids_clean_eval": ids_clean, "detector_threshold": threshold,
               "detector_fpr_clean_eval": eval_fpr, "n_attack_samples": len(x_atk),
               "attack_steps": config.ATTACK_STEPS, "lambdas": list(config.ADAPTIVE_LAMBDAS)}
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_results(results, config.RESULTS_DIR / "attack_results.png")

    torch.save(ids.state_dict(), config.RESULTS_DIR / "ids.pt")
    torch.save(detector.state_dict(), config.RESULTS_DIR / "detector.pt")
    np.savez(config.RESULTS_DIR / "indices.npz", val_idx=val_idx, eval_idx=eval_idx, atk_idx=atk_idx,
             threshold=threshold)
    torch.save(adaptive_samples, config.RESULTS_DIR / "adaptive_samples.pt")
    print(f"[7] Selesai dalam {time.time() - start:.1f}s. Hasil disimpan di {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
