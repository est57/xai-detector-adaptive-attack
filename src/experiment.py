import time

from src.attacks import adaptive_attack, make_success_fn, run_attack
from src.metrics import attack_metrics
from src.models import predict


def freeze(model):
    """Membekukan parameter model agar tidak ikut dilatih."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def evaluate_attack(ids, detector, threshold, constraints, x_orig, x_adv, eps, name):
    """Menghitung metrik serangan dan memeriksa pelanggaran constraint."""
    row = {"attack": name, "eps": eps,
           **attack_metrics(predict(ids, x_adv), detector.score(x_adv), threshold)}
    row["mean_linf"] = float((x_adv - x_orig).abs().max(dim=1).values.mean())
    row["violations"] = sum(constraints.violations(x_adv, x_orig, eps).values())
    return row


def attack_suite(ids, detector, threshold, constraints, x_atk, eps_list, steps, lambdas):
    """Serangan non-adaptif dan adaptif (sweep lambda) untuk setiap epsilon."""
    ids_only = make_success_fn(ids)
    full_success = make_success_fn(ids, detector, threshold)
    rows = [{"attack": "none", "eps": 0.0,
             **attack_metrics(predict(ids, x_atk), detector.score(x_atk), threshold)}]
    lambda_rows, adaptive_samples = [], {}

    for eps in eps_list:
        t0 = time.time()
        x_adv, _ = run_attack(ids, x_atk, constraints, eps, steps, ids_only)
        rows.append(evaluate_attack(ids, detector, threshold, constraints, x_atk, x_adv, eps, "non_adaptive"))

        x_adv, _, per_lambda = adaptive_attack(ids, detector, x_atk, constraints, eps, steps,
                                               full_success, lambdas)
        rows.append(evaluate_attack(ids, detector, threshold, constraints, x_atk, x_adv, eps, "adaptive"))
        adaptive_samples[eps] = x_adv
        lambda_rows += [{"eps": eps, "lambda": lam, "full_evasion_rate": r} for lam, r in per_lambda.items()]

        for row in rows[-2:]:
            print(f"  eps={eps:<5} {row['attack']:<13} ids_evasion={row['ids_evasion_rate']:.4f}  "
                  f"detected={row['detector_detection_rate']:.4f}  "
                  f"full_evasion={row['full_evasion_rate']:.4f}  violations={row['violations']}")
        print("  per lambda: " + "  ".join(f"{k}={v:.4f}" for k, v in per_lambda.items())
              + f"  ({time.time() - t0:.1f}s)")

    return rows, lambda_rows, adaptive_samples
