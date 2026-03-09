from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import average_precision_score, roc_curve

def precision_at_k(y_true, y_score, k: int) -> float:
    k = int(min(k, len(y_true)))
    if k <= 0:
        return 0.0
    # faster than full sort
    idx = np.argpartition(y_score, -k)[-k:]
    return float(np.mean(y_true[idx]))


def fraud_found_at_k(y_true, y_score, k: int) -> int:
    k = int(min(k, len(y_true)))
    if k <= 0:
        return 0
    idx = np.argpartition(y_score, -k)[-k:]
    return int(np.sum(y_true[idx]))


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(y_true, y_score, target_fpr: float) -> float:
    """
    Best achievable recall (TPR) subject to FPR <= target_fpr.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr
    return float(np.max(tpr[ok]) if np.any(ok) else 0.0)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    target_fpr: float
    achieved_fpr: float
    achieved_recall: float


def pick_threshold_by_fpr(y_true, y_score, target_fpr: float) -> OperatingPoint:
    """
    Choose a single threshold on validation data so that FPR <= target_fpr,
    and recall is maximised under that constraint.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)  # thr aligned with fpr/tpr
    ok = fpr <= target_fpr
    if not np.any(ok):
        # impossible constraint (or degenerate); pick strictest threshold
        return OperatingPoint(threshold=float("inf"), target_fpr=target_fpr,
                              achieved_fpr=0.0, achieved_recall=0.0)
    best = np.argmax(tpr[ok])
    thr_ok = thr[ok]
    fpr_ok = fpr[ok]
    tpr_ok = tpr[ok]
    return OperatingPoint(
        threshold=float(thr_ok[best]),
        target_fpr=float(target_fpr),
        achieved_fpr=float(fpr_ok[best]),
        achieved_recall=float(tpr_ok[best]),
    )


def metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold_precision": float(precision),
        "threshold_recall": float(recall),
        "threshold_fpr": float(fpr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def measure_latency_ms_per_row(model, X, warmup: int = 1, repeats: int = 3) -> dict:
    """
    CPU inference latency using predict_proba on provided matrix X.
    """
    # warmup
    for _ in range(max(warmup, 0)):
        _ = model.predict_proba(X)[:, 1]

    times = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        _ = model.predict_proba(X)[:, 1]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    sec = float(np.median(times))
    n = len(X)
    ms_per_row = (sec / max(n, 1)) * 1000.0
    rows_per_s = (n / sec) if sec > 0 else 0.0
    return {"ms_per_row": float(ms_per_row), "rows_per_s": float(rows_per_s)}