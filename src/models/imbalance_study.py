# src/models/imbalance_study.py
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.utils.split_manager import project_root, get_or_create_split
from src.utils.run_logger import append_run_csv
from src.utils.eval_metrics import (
    pr_auc, recall_at_fpr, precision_at_k, fraud_found_at_k,
    pick_threshold_by_fpr, metrics_at_threshold, measure_latency_ms_per_row
)

# imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = project_root()
SEED = 42


def compute_metrics(y_true, y_score, k=1000, target_fpr=0.01):
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "recall_at_fpr": recall_at_fpr(y_true, y_score, target_fpr=target_fpr),
        "precision_at_k": precision_at_k(y_true, y_score, k=k),
        "fraud_found_at_k": fraud_found_at_k(y_true, y_score, k=k),
    }


def cv_on_train(X_train, y_train, make_model_fn, k=1000, target_fpr=0.01, folds=3, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pr_list, r_list, p_list = [], [], []

    for tr_i, va_i in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[tr_i], y_train[tr_i]
        X_va, y_va = X_train.iloc[va_i], y_train[va_i]

        model = make_model_fn()
        model.fit(X_tr, y_tr)
        va_score = model.predict_proba(X_va)[:, 1]

        m = compute_metrics(y_va, va_score, k=k, target_fpr=target_fpr)
        pr_list.append(m["pr_auc"])
        r_list.append(m["recall_at_fpr"])
        p_list.append(m["precision_at_k"])

    return {
        "cv_folds": folds,
        "cv_pr_auc_mean": float(np.mean(pr_list)),
        "cv_recall_at_fpr_mean": float(np.mean(r_list)),
        "cv_precision_at_k_mean": float(np.mean(p_list)),
    }


def make_logreg_cost(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=None
        ))
    ])


def make_logreg_undersample(seed, under_ratio=0.2):
    # undersample majority to ratio (fraud : non-fraud roughly)
    rus = RandomUnderSampler(sampling_strategy=under_ratio, random_state=seed)
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("rus", rus),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=seed,
            n_jobs=None
        ))
    ])


def make_logreg_smote(seed, smote_ratio=0.2):
    sm = SMOTE(sampling_strategy=smote_ratio, random_state=seed)
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", sm),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=seed,
            n_jobs=None
        ))
    ])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--strategies", default="cost,undersample,smote")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--smote_ratio", type=float, default=0.2)
    ap.add_argument("--under_ratio", type=float, default=0.2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config, "r", encoding="utf-8"))

    dataset_path = (ROOT / cfg["dataset_path"]).resolve()
    seed = int(cfg.get("random_state", SEED))

    baseline_split_path = (ROOT / cfg.get("split_baseline_path", cfg.get("split_path", "results/splits/split.npz"))).resolve()
    time_split_path = (ROOT / cfg.get("split_time_path", "results/splits/split_time.npz")).resolve()
    train_on = cfg.get("train_on_split", "baseline")
    time_col = cfg.get("time_col")
    id_col = cfg.get("id_col")

    k = int(cfg.get("k", 1000))
    target_fpr = float(cfg.get("target_fpr", 0.01))

    df = pd.read_csv(dataset_path)
    y = df[cfg["target_col"]].astype(int).values

    drop_cols = [cfg["target_col"]] + cfg.get("drop_cols", [])
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).fillna(0)

    ids = df[id_col].astype(str).values if (id_col and id_col in df.columns) else None

    # splits (create both; pick one)
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

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    makers = {
        "cost": lambda: make_logreg_cost(seed),
        "undersample": lambda: make_logreg_undersample(seed, under_ratio=args.under_ratio),
        "smote": lambda: make_logreg_smote(seed, smote_ratio=args.smote_ratio),
    }

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    for strat in strategies:
        if strat not in makers:
            raise ValueError(f"Unknown strategy '{strat}'. Use: cost, undersample, smote")

        make_model_fn = makers[strat]

        # CV inside TRAIN only (leakage-safe)
        cv_summary = cv_on_train(
            X_train, y_train, make_model_fn,
            k=k, target_fpr=target_fpr, folds=args.folds, seed=seed
        )

        # Fit on full TRAIN split
        model = make_model_fn()
        model.fit(X_train, y_train)

        train_score = model.predict_proba(X_train)[:, 1]
        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]

        # main metrics
        train_m = compute_metrics(y_train, train_score, k=k, target_fpr=target_fpr)
        val_m = compute_metrics(y_val, val_score, k=k, target_fpr=target_fpr)
        test_m = compute_metrics(y_test, test_score, k=k, target_fpr=target_fpr)

        # operating point chosen on VAL, applied unchanged to TEST
        op = pick_threshold_by_fpr(y_val, val_score, target_fpr=target_fpr)
        val_op = metrics_at_threshold(y_val, val_score, op.threshold)
        test_op = metrics_at_threshold(y_test, test_score, op.threshold)

        # latency (test)
        lat = measure_latency_ms_per_row(model, X_test, warmup=1, repeats=3)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = f"results/metrics/logreg_{strat}_{run_id}.yaml"
        model_path = f"results/models/logreg_{strat}_{run_id}.joblib"

        metrics = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": f"logreg_{strat}",
            "split": "70/15/15 (train/val/test)",
            "seed": seed,
            "split_used": split_used,
            "split_file": split_file,
            "k": k,
            "target_fpr": target_fpr,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "fraud_rate_train": float(np.mean(y_train)),
            "fraud_rate_val": float(np.mean(y_val)),
            "fraud_rate_test": float(np.mean(y_test)),
            "fraud_count_train": int(np.sum(y_train)),
            "fraud_count_val": int(np.sum(y_val)),
            "fraud_count_test": int(np.sum(y_test)),
            **cv_summary,
            "train_pr_auc": train_m["pr_auc"],
            "train_recall_at_fpr": train_m["recall_at_fpr"],
            "train_precision_at_k": train_m["precision_at_k"],
            "train_fraud_found_at_k": train_m["fraud_found_at_k"],
            "val_pr_auc": val_m["pr_auc"],
            "val_recall_at_fpr": val_m["recall_at_fpr"],
            "val_precision_at_k": val_m["precision_at_k"],
            "val_fraud_found_at_k": val_m["fraud_found_at_k"],
            "test_pr_auc": test_m["pr_auc"],
            "test_recall_at_fpr": test_m["recall_at_fpr"],
            "test_precision_at_k": test_m["precision_at_k"],
            "test_fraud_found_at_k": test_m["fraud_found_at_k"],
            "op_threshold": op.threshold,
            "op_val_fpr": op.achieved_fpr,
            "op_val_recall": op.achieved_recall,
            **{f"val_op_{k}": v for k, v in val_op.items()},
            **{f"test_op_{k}": v for k, v in test_op.items()},
            "test_ms_per_row": lat["ms_per_row"],
            "test_rows_per_s": lat["rows_per_s"],
            "model_path": model_path,
            "metrics_path": metrics_path,
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        joblib.dump(model, model_path)
        append_run_csv(metrics)

        print(f"{strat} done -> {metrics_path}")

if __name__ == "__main__":
    main()