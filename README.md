# Evading Explanation-Based Adversarial Detectors: Domain-Constrained Adaptive Attacks on ML-IDS

[English](#english) | [Bahasa Indonesia](#bahasa-indonesia)

## English

This repository contains the experiment code for the article *Evading Explanation-Based Adversarial Detectors: Domain-Constrained Adaptive Attacks on Machine Learning Intrusion Detection Systems* (M. C. Anwar and H. Prayoga, Universitas Bina Sarana Informatika). It evaluates an adversarial example detector that reads Integrated Gradients (IG) attributions of an MLP intrusion detection system (IDS) against three attacks, all restricted by the domain constraints of network traffic:

- **Non-adaptive**: constrained PGD against the IDS only.
- **ASA-style**: constrained PGD that pulls attributions toward the benign attribution centroid, without knowledge of the detector.
- **Adaptive**: constrained PGD against the IDS and the detector together, with gradients through the IG computation.

A defense experiment then retrains the detector with adaptive samples for five rounds and tests it with transferred and regenerated attacks. Experiments use NSL-KDD and UNSW-NB15 with five random seeds.

### Repository structure

| File | Purpose |
|---|---|
| `config.py` | All experiment parameters (epochs, budgets, attack steps, seeds) |
| `src/datasets.py` | Dataset specifications for NSL-KDD and UNSW-NB15: files, columns, log features, domain constraints |
| `src/data.py` | Loading, binary labels, one-hot encoding, log1p, min-max scaling |
| `src/constraints.py` | Domain constraints: immutable, increase-only, range [0, 1], integer, derived features |
| `src/models.py` | MLP model and training |
| `src/explain.py` | Differentiable Integrated Gradients |
| `src/detector.py` | Attribution-based adversarial example detector |
| `src/attacks.py` | Constrained PGD (non-adaptive, adaptive, ASA-style) |
| `src/metrics.py` | IDS metrics, detector threshold calibration, attack metrics |
| `src/plots.py` | Result figures |
| `src/experiment.py` | Shared helpers: model freezing, attack evaluation, attack suite per budget |
| `run_experiment.py` | Main experiment: attacks against the initial detector |
| `run_defense.py` | Defense experiment: detector retraining with adaptive samples |
| `run_baseline.py` | ASA-style baseline against the initial detector |
| `run_seeds.sh` | Runs several seeds in parallel and aggregates them |
| `aggregate.py` | Mean and standard deviation across seeds, error-bar figure |
| `make_figures.py` | Regenerates per-seed figures and the article figures (`results/figures/`) from the result CSV files |

### Pipeline

1. Load and preprocess the dataset (`--dataset nsl_kdd` or `--dataset unsw_nb15`). The official test set is split, stratified by label, into 30% validation data (threshold calibration) and 70% evaluation data.
2. Train the MLP IDS (128-64, SiLU) and evaluate it on clean evaluation data.
3. Generate non-adaptive adversarial samples from the training data.
4. Train the detector: input -> IG attributions -> clean/adversarial classifier. The detector score is the logit margin (adversarial minus clean logit), and the threshold is calibrated to a 5% false positive rate on validation data.
5. Select attack samples from the evaluation data that the IDS classifies correctly.
6. Run the attacks for ε ∈ {0.02, 0.05, 0.1, 0.2, 0.3} with 200 PGD steps, best-iterate selection, and a sweep over λ ∈ {0.3, 1, 3, 10}.
7. Save the results to `results/<dataset>/seed_<seed>/`.

The defense experiment (`run_defense.py`) needs the outputs of `run_experiment.py` (`ids.pt`, `detector.pt`, `indices.npz`, `adaptive_samples.pt`). The IDS is never retrained; only the detector is.

### Domain constraints

| Constraint | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Immutable | protocol_type, service, flag, content features | proto, service, state, TTL, TCP window, server-side and application-layer features |
| Increase-only | duration, src_bytes, dst_bytes | dur, spkts, sbytes, sjit, seven ct_* features |
| Free within range | time- and host-based traffic features | – |
| Integer | duration, src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count | spkts, sbytes, mutable ct_* features |
| Derived (recomputed) | – | smean, sload, rate, sinpkt |

Derived features are updated as `v + h(p_new) - h(p_old)`, so unperturbed samples stay unchanged. Each added packet is limited to 40–1500 bytes.

### Attack objectives

- Non-adaptive: `min CE(IDS(x'), normal)`
- Adaptive: `min CE(IDS(x'), normal) + lambda * CE(Detector(IG(x')), clean)`
- ASA-style: `min CE(IDS(x'), normal) + mu * D(x')`, where `D` is the standardized squared distance between the attribution and the benign attribution centroid

### Metrics

- `ids_evasion_rate`: share of attack samples classified as normal by the IDS.
- `detector_detection_rate`: share of samples flagged as adversarial by the detector.
- `full_evasion_rate`: share of samples that evade the IDS and are not flagged (main metric).
- `violations`: number of samples that break a domain constraint (must be 0).

### Main results (ε = 0.3, mean ± SD over five seeds)

| Full evasion rate | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Non-adaptive | 13.8 ± 2.6% | 3.2 ± 1.2% |
| ASA-style | 23.6 ± 3.0% | 0.7 ± 0.5% |
| Adaptive | 68.1 ± 13.0% | 30.8 ± 4.2% |
| Adaptive, regenerated after 5 retraining rounds | 77.5 ± 5.3% | 22.1 ± 3.9% |

### Datasets

| Dataset | Files | Source |
|---|---|---|
| NSL-KDD | `data/KDDTrain+.txt`, `data/KDDTest+.txt` | GitHub mirror defcom17/NSL_KDD |
| UNSW-NB15 | `data/unsw_nb15/UNSW_NB15_training-set.csv`, `..._testing-set.csv` | GitHub mirror tylerhoward15/tensorFlow-MLP-UNSW-NB15 (175,341 / 82,332 records, matching the official counts) |

The datasets are not included in this repository; download them into `data/` before running the experiments.

### Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py --dataset unsw_nb15 --seed 42
.venv/bin/python run_defense.py --dataset unsw_nb15 --seed 42
.venv/bin/python run_baseline.py --dataset unsw_nb15 --seed 42
```

All five seeds in parallel, followed by aggregation:

```bash
DATASET=unsw_nb15 ./run_seeds.sh 42 7 123 2024 31337
```

Aggregation only, and regeneration of all figures from the saved CSV files:

```bash
.venv/bin/python aggregate.py --dataset unsw_nb15
.venv/bin/python make_figures.py
```

Logs are written to `logs/`, aggregated results to `results/<dataset>/aggregate/`, and article figures to `results/figures/`.

## Bahasa Indonesia

### Struktur

| File | Fungsi |
|---|---|
| `config.py` | Seluruh parameter eksperimen (epoch, epsilon, jumlah langkah serangan, dll.) |
| `src/datasets.py` | Spesifikasi dataset (NSL-KDD, UNSW-NB15): file, kolom, fitur log, dan batasan domain |
| `src/data.py` | Memuat dataset, label biner, one-hot, log1p, min-max scaling |
| `src/constraints.py` | Batasan domain: immutable, increase-only, rentang [0, 1], integer, fitur turunan |
| `src/models.py` | Model MLP dan fungsi pelatihan |
| `src/explain.py` | Integrated Gradients (dapat diturunkan untuk serangan adaptif) |
| `src/detector.py` | Detektor adversarial berbasis atribusi penjelasan |
| `src/attacks.py` | Constrained PGD: non-adaptif dan adaptif |
| `src/metrics.py` | Metrik IDS, kalibrasi threshold detektor, metrik serangan |
| `src/plots.py` | Grafik hasil |
| `src/experiment.py` | Fungsi bersama: membekukan model, evaluasi serangan, rangkaian serangan per epsilon |
| `run_experiment.py` | Eksperimen utama: serangan terhadap detektor awal (langkah 1 sampai 7) |
| `run_defense.py` | Eksperimen pertahanan: detektor dilatih ulang dengan sampel adaptif (langkah 1 sampai 6) |
| `run_baseline.py` | Pembanding: serangan bergaya ASA-IDS (atribusi ditarik ke pusat atribusi benign, tanpa pengetahuan detektor) terhadap detektor awal |
| `run_seeds.sh` | Menjalankan beberapa seed secara paralel lalu agregasi |
| `aggregate.py` | Rata-rata dan simpangan baku antar seed, grafik error bar |
| `make_figures.py` | Membuat ulang grafik per seed dan grafik artikel (`results/figures/`) dari file CSV hasil uji coba |

### Alur eksperimen

1. Muat dan proses dataset (`--dataset nsl_kdd` atau `--dataset unsw_nb15`). Data uji resmi dibagi stratified: 30% validasi (kalibrasi threshold), 70% evaluasi.
2. Latih model IDS (MLP) dan evaluasi pada data evaluasi bersih.
3. Buat sampel adversarial non-adaptif dari data latih.
4. Latih detektor: input -> atribusi Integrated Gradients -> classifier bersih/adversarial. Skor detektor = logit adversarial - logit bersih (logit margin, tidak jenuh seperti probabilitas softmax). Threshold skor dikalibrasi pada FPR 5% di data validasi.
5. Pilih sampel serangan dari data evaluasi yang terdeteksi benar oleh IDS.
6. Jalankan serangan non-adaptif (hanya menargetkan IDS) dan adaptif (menargetkan IDS + detektor, sweep lambda) untuk beberapa epsilon.
7. Simpan hasil ke `results/`.

### Penguatan serangan

- Best-iterate: setiap iterasi, kandidat difinalisasi (proyeksi + pembulatan integer) lalu diperiksa; iterasi pertama yang berhasil disimpan per sampel.
- Sweep lambda (`ADAPTIVE_LAMBDAS`): serangan diulang untuk beberapa lambda, per sampel diambil hasil yang berhasil.
- Jumlah langkah PGD: 200.

### Eksperimen pertahanan (`run_defense.py`)

Membutuhkan output `run_experiment.py` (`ids.pt`, `detector.pt`, `indices.npz`, `adaptive_samples.pt`).

1. Muat dataset, model IDS, detektor awal, dan indeks data.
2. Siapkan data latih bersih dan sampel non-adaptif.
3. Sebanyak `DEFENSE_ROUNDS` ronde: buat sampel adaptif terhadap detektor saat ini, tambahkan ke data latih, latih detektor baru, kalibrasi ulang threshold. Setelah setiap ronde dijalankan probe serangan adaptif (`PROBE_EPS`, `PROBE_LAMBDAS`) pada data evaluasi; hasilnya di `defense_rounds.csv` dan `defense_rounds.png`.
4. Uji transfer: sampel adaptif yang dibuat untuk detektor awal diuji ke detektor baru.
5. Serangan adaptif baru terhadap detektor yang sudah dilatih ulang.
6. Simpan `defense_results.csv`, `defense_lambda_results.csv`, `defense_summary.json`, `defense_results.png`.

Model IDS tidak dilatih ulang; pertahanan hanya pada detektor.

### Dataset

| Dataset | File | Sumber |
|---|---|---|
| NSL-KDD | `data/KDDTrain+.txt`, `data/KDDTest+.txt` | mirror GitHub defcom17/NSL_KDD |
| UNSW-NB15 | `data/unsw_nb15/UNSW_NB15_training-set.csv`, `..._testing-set.csv` | mirror GitHub tylerhoward15/tensorFlow-MLP-UNSW-NB15 (175.341 / 82.332 baris, sesuai angka resmi UNSW) |

### Batasan domain NSL-KDD

| Jenis | Fitur |
|---|---|
| Immutable | protocol_type, service, flag, fitur konten (hot, logged_in, root_shell, dll.) |
| Increase-only | duration, src_bytes, dst_bytes (penyerang hanya bisa menambah padding/jeda) |
| Bebas dalam rentang | fitur trafik berbasis waktu dan host (count, *_rate, dst_host_*) |
| Integer | duration, src_bytes, dst_bytes, count, srv_count, dst_host_count, dst_host_srv_count |

### Batasan domain UNSW-NB15

| Jenis | Fitur |
|---|---|
| Immutable | proto, service, state, TTL, window TCP, fitur sisi server (dpkts, dbytes, dload, ...), fitur aplikasi (trans_depth, ct_flw_http_mthd, ...) |
| Increase-only | dur, spkts, sbytes, sjit, ct_srv_src, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_src_ltm, ct_srv_dst |
| Integer | spkts, sbytes, semua fitur ct_* yang mutable |
| Turunan (dihitung ulang) | smean = sbytes/spkts, sload = (sbytes - smean) * 8/dur, rate = (spkts + dpkts - 1)/dur, sinpkt = dur * 1000/(spkts - 1) |
| Konsistensi paket | tiap paket tambahan minimal 40 byte, maksimal 1500 byte per paket |

Fitur turunan dihitung sebagai `nilai_asli + g(fitur_baru) - g(fitur_asli)` sehingga sampel tanpa perturbasi tidak berubah.

### Fungsi objektif serangan

- Non-adaptif: `min CE(IDS(x'), normal)`
- Adaptif: `min CE(IDS(x'), normal) + lambda * CE(Detektor(IG(x')), bersih)`

### Metrik

- `ids_evasion_rate`: proporsi sampel serangan yang diklasifikasikan normal oleh IDS.
- `detector_detection_rate`: proporsi sampel yang ditandai adversarial oleh detektor.
- `full_evasion_rate`: lolos dari IDS dan tidak terdeteksi detektor (metrik utama).
- `violations`: jumlah sampel yang melanggar batasan domain (harus 0).

### Output

Semua hasil ada di `results/<dataset>/seed_<seed>/`:

- `attack_results.csv` dan `attack_results.png`: hasil utama per epsilon.
- `lambda_results.csv`: full evasion rate untuk setiap lambda.
- `summary.json`: performa IDS, threshold, dan FPR detektor.
- `defense_*.csv`, `defense_rounds.png`: hasil eksperimen pertahanan.

### Menjalankan

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py --dataset unsw_nb15 --seed 42
.venv/bin/python run_defense.py --dataset unsw_nb15 --seed 42
```

Beberapa seed sekaligus (paralel), lalu agregasi rata-rata dan simpangan baku:

```bash
DATASET=unsw_nb15 ./run_seeds.sh 42 7 123 2024 31337
```

Pembanding bergaya ASA-IDS (butuh output `run_experiment.py` pada seed yang sama), lalu agregasi ulang:

```bash
.venv/bin/python run_baseline.py --dataset unsw_nb15 --seed 42
```

Log ada di `logs/`, agregasi di `results/<dataset>/aggregate/` (`main_mean_std.csv`, `defense_mean_std.csv`, `rounds_mean_std.csv`, `ids_mean_std.csv`, `aggregate.png`). Agregasi saja: `.venv/bin/python aggregate.py --dataset unsw_nb15`.
