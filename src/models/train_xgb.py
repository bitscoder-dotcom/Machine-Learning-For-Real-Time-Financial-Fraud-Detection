import os
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
from src.utils.run_logger import append_run_csv
from src.utils.split_manager import project_root, get_or_create_split

ROOT = project_root()
SEED = 42


def recall_at_fixed_precision(y_true, y_score, min_precision=0.10) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = precision >= min_precision
    return float(np.max(recall[valid])) if np.any(valid) else 0.0


def precision_at_k(y_true, y_score, k=1000) -> float:
    k = min(k, len(y_true))
    topk_idx = np.argsort(y_score)[-k:]
    return float(np.mean(y_true[topk_idx]))


def fraud_found_at_k(y_true, y_score, k=1000) -> int:
    k = min(k, len(y_true))
    topk_idx = np.argsort(y_score)[-k:]
    return int(np.sum(y_true[topk_idx]))


def compute_split_metrics(y_true, y_score, k=1000) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision_0.10": float(recall_at_fixed_precision(y_true, y_score, 0.10)),
        "recall_at_precision_0.20": float(recall_at_fixed_precision(y_true, y_score, 0.20)),
        "precision_at_1000": float(precision_at_k(y_true, y_score, k=k)),
        "fraud_found_at_1000": int(fraud_found_at_k(y_true, y_score, k=k)),
    }


def main():
    with open(ROOT / "configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_state", SEED))
    k = 1000

    dataset_path = (ROOT / cfg["dataset_path"]).resolve()
    split_path = (ROOT / cfg.get("split_path", "results/splits/split.npz")).resolve()

    df = pd.read_csv(dataset_path)
    y = df[cfg["target_col"]].astype(int).values
    X = df.drop(columns=[cfg["target_col"]]).select_dtypes(include=[np.number]).fillna(0)

    train_idx, val_idx, test_idx = get_or_create_split(
        y=y,
        seed=seed,
        split_path=split_path,
        dataset_path=dataset_path,
    )

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]


    # imbalance ratio for scale_pos_weight
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / max(n_pos, 1))

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)

    train_score = model.predict_proba(X_train)[:, 1]
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    print("MODEL TYPE:", type(model))
    print("BOOSTED ROUNDS:", model.get_booster().num_boosted_rounds())
    print("VAL SCORE HEAD:", val_score[:10])

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "xgb_baseline",
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
        "scale_pos_weight": float(scale_pos_weight),
    }

    train_m = compute_split_metrics(y_train, train_score, k=k)
    val_m = compute_split_metrics(y_val, val_score, k=k)
    test_m = compute_split_metrics(y_test, test_score, k=k)

    metrics.update({f"train_{key}": value for key, value in train_m.items()})
    metrics.update({f"val_{key}": value for key, value in val_m.items()})
    metrics.update({f"test_{key}": value for key, value in test_m.items()})

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"results/metrics/xgb_{run_id}.yaml"
    model_path = f"results/models/xgb_{run_id}.joblib"

    metrics["model_path"] = model_path
    metrics["metrics_path"] = metrics_path

    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    print("Saving model type:", type(model))
    assert "XGBClassifier" in str(type(model)), "Not saving an XGBClassifier model!"

    joblib.dump(model, model_path)
    append_run_csv(metrics)

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved model   -> {model_path}")
    print("Appended run  -> results/metrics/runs.csv")
    print(metrics)


if __name__ == "__main__":
    main()
