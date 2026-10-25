#!/usr/bin/env bash
# Menjalankan eksperimen utama dan pertahanan untuk beberapa seed secara paralel, lalu agregasi.
# Contoh: DATASET=unsw_nb15 ./run_seeds.sh 42 7 123 2024 31337
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
THREADS=${THREADS:-2}
DATASET=${DATASET:-nsl_kdd}
SEEDS=("$@")
mkdir -p logs

for seed in "${SEEDS[@]}"; do
  (
    $PY -u run_experiment.py --dataset "$DATASET" --seed "$seed" --threads "$THREADS" > "logs/${DATASET}_experiment_seed_${seed}.log" 2>&1
    $PY -u run_defense.py --dataset "$DATASET" --seed "$seed" --threads "$THREADS" > "logs/${DATASET}_defense_seed_${seed}.log" 2>&1
    echo "seed $seed selesai"
  ) &
done
wait

$PY aggregate.py --dataset "$DATASET"
