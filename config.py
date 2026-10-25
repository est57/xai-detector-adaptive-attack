import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "results"
RESULTS_DIR = RESULTS_ROOT
DATASET_DIR = RESULTS_ROOT

DATASET = "nsl_kdd"

SEED = 42
SEEDS = (42, 7, 123, 2024, 31337)
THREADS = 8
DEVICE = "cpu"

# Model IDS
IDS_HIDDEN = (128, 64)
IDS_EPOCHS = 15
IDS_LR = 1e-3
BATCH_SIZE = 256

# Penjelasan (Integrated Gradients)
IG_STEPS = 16

# Detektor berbasis penjelasan
DET_HIDDEN = (64, 32)
DET_EPOCHS = 20
DET_LR = 1e-3
DET_TRAIN_EPS = (0.02, 0.05, 0.1, 0.2)
DET_TRAIN_SAMPLES = 4000
DET_TARGET_FPR = 0.05

VAL_FRACTION = 0.3

# Serangan
ATTACK_STEPS = 200
ATTACK_EPS_LIST = (0.02, 0.05, 0.1, 0.2, 0.3)
ADAPTIVE_LAMBDAS = (0.3, 1.0, 3.0, 10.0)
MAX_ATTACK_SAMPLES = 3000

# Pertahanan: pelatihan ulang detektor dengan sampel adaptif
DEFENSE_ROUNDS = 5
DEFENSE_LAMBDA = 1.0
PROBE_EPS = 0.3
PROBE_LAMBDAS = (0.3, 1.0)

# Pembanding: serangan bergaya ASA-IDS (menarik atribusi ke pusat atribusi benign)
STEALTH_MUS = (0.3, 1.0, 3.0, 10.0)


def setup(argv=None, make_dirs=True):
    """Membaca argumen --dataset, --seed, dan --threads, lalu mengatur folder hasil."""
    global DATASET, SEED, THREADS, DATASET_DIR, RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15"], default=DATASET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args(argv)
    DATASET, SEED, THREADS = args.dataset, args.seed, args.threads
    DATASET_DIR = RESULTS_ROOT / DATASET
    RESULTS_DIR = DATASET_DIR / f"seed_{SEED}"
    if make_dirs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
