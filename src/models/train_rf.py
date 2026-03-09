# src/models/train_rf.py
import os, yaml, joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from src.utils.run_logger import append_run_csv
from src.utils.split_manager import project_root, get_or_create_split
from src.utils.eval_metrics import (
    pr_auc, recall_at_fpr, precision_at_k, fraud_found_at_k,
    pick_threshold_by_fpr, metrics_at_threshold, measure_latency_ms_per_row
)

ROOT = project_root()
SEED = 42

def compute_split_metrics(y_true, y_score, k=1000, target_fpr=0.01) -> dict:
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "recall_at_fpr": recall_at_fpr(y_true, y_score, target_fpr=target_fpr),
        "precision_at_k": precision_at_k(y_true, y_score, k=k),
        "fraud_found_at_k": fraud_found_at_k(y_true, y_score, k=k),
    }

def main():
    # accept same --config pattern you already added in other scripts
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = (ROOT / cfg["dataset_path"]).resolve()
    seed = int(cfg.get("random_state", SEED))

    baseline_split_path = (ROOT / cfg.get("split_baseline_path", cfg.get("split_path", "results/splits/split.npz"))).resolve()
    time_split_path = (ROOT / cfg.get("split_time_path", "results/splits/split_time.npz")).resolve()

    train_on = cfg.get("train_on_split", "baseline")
    time_col = cfg.get("time_col")
    id_col = cfg.get("id_col")

    df = pd.read_csv(dataset_path)
    y = df[cfg["target_col"]].astype(int).values

    drop_cols = [cfg["target_col"]] + cfg.get("drop_cols", [])
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).fillna(0)

    ids = df[id_col].astype(str).values if (id_col and id_col in df.columns) else None

    # splits
    tr_b, va_b, te_b = get_or_create_split(
        y=y, seed=seed, split_path=baseline_split_path, dataset_path=dataset_path,
        method="stratified", ids_for_overlap_check=ids
    )

    tr_t = va_t = te_t = None
    if time_col and time_col in df.columns:
        tr_t, va_t, te_t = get_or_create_split(
            y=y, seed=seed, split_path=time_split_path, dataset_path=dataset_path,
            method="time", t=df[time_col].values, ids_for_overlap_check=ids
        )

    if train_on == "time" and tr_t is not None:
        train_idx, val_idx, test_idx = tr_t, va_t, te_t
        split_used = f"time({time_col})"
        split_file = str(time_split_path)
    else:
        train_idx, val_idx, test_idx = tr_b, va_b, te_b
        split_used = "baseline_stratified"
        split_file = str(baseline_split_path)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    k = int(cfg.get("k", 1000))
    target_fpr = float(cfg.get("target_fpr", 0.01))

    rf_cfg = cfg.get("rf", {})
    model = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=seed,
        class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
    )

    model.fit(X_train, y_train)

    train_score = model.predict_proba(X_train)[:, 1]
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "rf_baseline",
        "split": "70/15/15 (train/val/test)",
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "fraud_rate_train": float(np.mean(y_train)),
        "fraud_rate_val": float(np.mean(y_val)),
        "fraud_rate_test": float(np.mean(y_test)),
        "fraud_count_train": int(np.sum(y_train)),
        "fraud_count_val": int(np.sum(y_val)),
        "fraud_count_test": int(np.sum(y_test)),
        "split_used": split_used,
        "split_file": split_file,
        "k": k,
        "target_fpr": target_fpr,
    }

    # split-level metrics
    metrics.update({f"train_{k}": v for k, v in compute_split_metrics(y_train, train_score, k=k, target_fpr=target_fpr).items()})
    metrics.update({f"val_{k}": v for k, v in compute_split_metrics(y_val, val_score, k=k, target_fpr=target_fpr).items()})
    metrics.update({f"test_{k}": v for k, v in compute_split_metrics(y_test, test_score, k=k, target_fpr=target_fpr).items()})

    # operating point on val, applied to test
    op = pick_threshold_by_fpr(y_val, val_score, target_fpr=target_fpr)
    metrics["op_threshold"] = op.threshold
    metrics["op_val_fpr"] = op.achieved_fpr
    metrics["op_val_recall"] = op.achieved_recall
    metrics.update({f"val_{k}": v for k, v in metrics_at_threshold(y_val, val_score, op.threshold).items()})
    metrics.update({f"test_{k}": v for k, v in metrics_at_threshold(y_test, test_score, op.threshold).items()})

    # latency
    lat = measure_latency_ms_per_row(model, X_test, warmup=1, repeats=3)
    metrics["test_ms_per_row"] = lat["ms_per_row"]
    metrics["test_rows_per_s"] = lat["rows_per_s"]

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"results/metrics/rf_{run_id}.yaml"
    model_path = f"results/models/rf_{run_id}.joblib"
    metrics["model_path"] = model_path
    metrics["metrics_path"] = metrics_path

    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    joblib.dump(model, model_path)
    append_run_csv(metrics)

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved model   -> {model_path}")
    print(metrics)

if __name__ == "__main__":
    main()