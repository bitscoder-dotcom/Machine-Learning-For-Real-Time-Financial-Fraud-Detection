import os
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier

SEED = 42

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve


def recall_at_fixed_precision(y_true, y_score, min_precision=0.10):
    """
    Best recall achievable while precision >= min_precision.
    Useful for fraud where you want to avoid flooding analysts with false alerts.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = precision >= min_precision
    return float(np.max(recall[valid]) if np.any(valid) else 0.0)

def precision_at_k(y_true, y_score, k=1000) -> float:
    """
    Precision at k: if you take the top-k highest-risk transactions, what fraction is
    actually fraud?
    """
    k = min(k, len(y_true))
    topK_idx = np.argsort(y_score)[-k:]
    return float(np.mean(y_true[topK_idx]))

def fraud_found_at_k(y_true, y_score, k=1000) -> int:
    """Number of fraud cases in the top-k highest-rank transactions."""
    k = min(k, len(y_true))
    topK_idx = np.argsort(y_score)[-k:]
    return int(np.sum(y_true[topK_idx]))

def stratified_split_70_15_15(X, y, seed=SEED):
    """
    Two-step split:
        - test = 15%
        - remaining 85% split into train (70%) and val (15%)
    """
    # step 1: test = 15%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )

    # step 2: val is 15% overall => 15/85 of the remaining set
    val_share_of_temp = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_share_of_temp, random_state=seed, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_split_metrics(y_true, y_score, k=1000) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision_0.10": float(recall_at_fixed_precision(y_true, y_score, min_precision=0.10)),
        "recall_at_precision_0.20": float(recall_at_fixed_precision(y_true, y_score, min_precision=0.20)),
        "precision_at_1000": float(precision_at_k(y_true, y_score, k=k)),
        "fraud_found_at_1000": int(fraud_found_at_k(y_true, y_score, k=k)),
    }

def append_run_csv(row: dict, path="results/metrics/runs.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)

def main():
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_state", SEED))
    df = pd.read_csv(cfg["dataset_path"])
    y = df[cfg["target_col"]].astype(int).values
    X = df.drop(columns=[cfg["target_col"]])
    k = 1000

    # Basic Handling: drop non-numeric cols if any
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # 70/15/15 split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_70_15_15(X, y, seed=seed)

    # Model pipeline: scale -> logistic regression with class weighting
    model = Pipeline(steps=[("scaler", StandardScaler()),
                            ("clf", LogisticRegression(
            max_iter=cfg["model"].get("max_iter", 2000),
            class_weight=cfg["model"].get("class_weight", "balanced"),
            n_jobs=None
                            ))
        ])

    model.fit(X_train, y_train)
    train_score = model.predict_proba(X_train)[:, 1]
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "logreg_balanced",
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
    }

    # Adding split metrics
    train_m = compute_split_metrics(y_train, train_score, k=k)
    val_m = compute_split_metrics(y_val, val_score, k=k)
    test_m = compute_split_metrics(y_test, test_score, k=k)

    metrics.update({f"train_{key}": value for key, value in train_m.items()})
    metrics.update({f"val_{key}": value for key, value in val_m.items()})
    metrics.update({f"test_{key}": value for key, value in test_m.items()})

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    metrics_path = "results/metrics/baseline_logreg.yaml"
    model_path = "results/models/baseline_logreg.joblib"

    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    joblib.dump(model, model_path)

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved models -> {model_path}")
    print("Appended run -> results/metrics/runs.csv")
    print(metrics)


if __name__ == "__main__":
    main()