import os
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve
from src.utils.run_logger import append_run_csv
from src.utils.split_manager import project_root, get_or_create_split

SEED = 42
ROOT = project_root()


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


def compute_split_metrics(y_true, y_score, k=1000) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision_0.10": float(recall_at_fixed_precision(y_true, y_score, min_precision=0.10)),
        "recall_at_precision_0.20": float(recall_at_fixed_precision(y_true, y_score, min_precision=0.20)),
        "precision_at_1000": float(precision_at_k(y_true, y_score, k=k)),
        "fraud_found_at_1000": int(fraud_found_at_k(y_true, y_score, k=k)),
    }

def main():
    with open(ROOT / "configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dataset_path = (ROOT / cfg["dataset_path"]).resolve()
    split_path = (ROOT / cfg.get("split_path", "results/splits/split.npz")).resolve()

    seed = int(cfg.get("random_state", SEED))
    df = pd.read_csv(dataset_path)
    y = df[cfg["target_col"]].astype(int).values
    X = df.drop(columns=[cfg["target_col"]]).select_dtypes(include=[np.number]).fillna(0)

    k = 1000

    train_idx, val_idx, test_idx = get_or_create_split(
        y=y,
        seed=seed,
        split_path=split_path,
        dataset_path=dataset_path,
    )

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"results/metrics/logreg_{run_id}.yaml"
    model_path = f"results/models/logreg_{run_id}.joblib"

    metrics["model_path"] = model_path
    metrics["metrics_path"] = metrics_path

    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    clf = model.named_steps["clf"]
    print("Saving model type:", type(model), " | clf:", type(clf))
    assert isinstance(clf, LogisticRegression), "Not saving a LogisticRegression classifier!"

    joblib.dump(model, model_path)
    append_run_csv(metrics)

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved models -> {model_path}")
    print("Appended run -> results/metrics/runs.csv")
    print(metrics)


if __name__ == "__main__":
    main()