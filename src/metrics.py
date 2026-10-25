import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def ids_metrics(y_true, prob, threshold=0.5):
    """Metrik performa model IDS pada data bersih."""
    y_pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, prob),
    }


def calibrate_threshold(clean_scores, target_fpr):
    """Threshold detektor agar false positive rate pada data bersih sama dengan target."""
    return float(np.quantile(clean_scores, 1 - target_fpr))


def attack_metrics(ids_prob, det_score, det_threshold):
    """Metrik serangan: evasion terhadap IDS, deteksi oleh detektor, dan evasion total."""
    evade_ids = ids_prob < 0.5
    flagged = det_score >= det_threshold
    return {
        "ids_evasion_rate": float(evade_ids.mean()),
        "detector_detection_rate": float(flagged.mean()),
        "detector_detection_on_evasive": float(flagged[evade_ids].mean()) if evade_ids.any() else float("nan"),
        "full_evasion_rate": float((evade_ids & ~flagged).mean()),
    }
