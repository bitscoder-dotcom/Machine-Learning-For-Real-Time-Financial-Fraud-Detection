import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.utils.split_manager import project_root, get_or_create_split


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


def topk_overlap(a, b, k=1000) -> float:
    k = min(k, len(a))
    ia = set(np.argsort(a)[-k:])
    ib = set(np.argsort(b)[-k:])
    return len(ia & ib) / k


def compute_metrics(y_true, y_score, k=1000) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision_0.10": float(recall_at_fixed_precision(y_true, y_score, 0.10)),
        "recall_at_precision_0.20": float(recall_at_fixed_precision(y_true, y_score, 0.20)),
        "precision_at_k": float(precision_at_k(y_true, y_score, k=k)),
        "fraud_found_at_k": int(fraud_found_at_k(y_true, y_score, k=k)),
    }


def measure_latency(model, X, n_rows=50000, n_repeats=3) -> dict:
    n_rows = min(int(n_rows), len(X))
    Xs = X.iloc[:n_rows]

    # warm-up (important for fair timing)
    _ = model.predict_proba(Xs.iloc[:200])[:, 1]

    times = []
    for _ in range(int(n_repeats)):
        t0 = time.perf_counter()
        _ = model.predict_proba(Xs)[:, 1]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_s = float(np.mean(times))
    ms_per_row = (avg_s * 1000.0) / n_rows
    rows_per_s = n_rows / avg_s if avg_s > 0 else float("inf")
    return {"latency_ms_per_row": float(ms_per_row), "throughput_rows_per_s": float(rows_per_s)}


def load_latest_models_from_runs(runs_path: Path, wanted_models: list[str], root: Path) -> dict:
    runs = pd.read_csv(runs_path)

    # parse timestamp (your CSV uses ISO like 2026-02-20T13:10:49)
    runs["timestamp"] = pd.to_datetime(runs["timestamp"], errors="coerce")

    latest = {}
    for m in wanted_models:
        sub = runs[runs["model"] == m].sort_values("timestamp")
        if sub.empty:
            raise FileNotFoundError(f"No runs found for model='{m}' in {runs_path}")
        row = sub.iloc[-1].to_dict()

        model_path = (root / str(row["model_path"])).resolve()
        metrics_path = (root / str(row["metrics_path"])).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        latest[m] = {"row": row, "model_path": model_path, "metrics_path": metrics_path}

    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--latency_rows", type=int, default=50000)
    ap.add_argument("--latency_repeats", type=int, default=3)
    ap.add_argument("--similarity", action="store_true", help="Print diff/spearman/topK overlap on val")
    args = ap.parse_args()

    root = project_root()

    cfg = yaml.safe_load(open(root / "configs/base.yaml", "r", encoding="utf-8"))
    dataset_path = (root / cfg["dataset_path"]).resolve()
    split_path = (root / cfg.get("split_path", "results/splits/split.npz")).resolve()
    runs_path = (root / "results/metrics/runs.csv").resolve()

    seed = int(cfg.get("random_state", SEED))

    df = pd.read_csv(dataset_path)
    y = df[cfg["target_col"]].astype(int).values
    X = df.drop(columns=[cfg["target_col"]]).select_dtypes(include=[np.number]).fillna(0)

    train_idx, val_idx, test_idx = get_or_create_split(
        y=y,
        seed=seed,
        split_path=split_path,
        dataset_path=dataset_path,
    )

    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    wanted = ["logreg_balanced", "xgb_baseline"]
    latest = load_latest_models_from_runs(runs_path, wanted, root)

    rows = []
    preds_val = {}

    for model_name, info in latest.items():
        model = joblib.load(info["model_path"])
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        preds_val[model_name] = p_val

        m_val = compute_metrics(y_val, p_val, k=args.k)
        m_test = compute_metrics(y_test, p_test, k=args.k)
        lat = measure_latency(model, X_test, n_rows=args.latency_rows, n_repeats=args.latency_repeats)

        rows.append({
            "model": model_name,
            "run_timestamp": str(info["row"].get("timestamp")),
            "val_pr_auc": m_val["pr_auc"],
            "test_pr_auc": m_test["pr_auc"],
            f"val_precision@{args.k}": m_val["precision_at_k"],
            f"val_fraud_found@{args.k}": m_val["fraud_found_at_k"],
            f"test_precision@{args.k}": m_test["precision_at_k"],
            f"test_fraud_found@{args.k}": m_test["fraud_found_at_k"],
            "test_ms_per_row": lat["latency_ms_per_row"],
            "test_rows_per_s": lat["throughput_rows_per_s"],
            "model_path": str(info["model_path"].relative_to(root)),
        })

    leaderboard = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    pd.set_option("display.max_colwidth", 120)
    print("\n=== Leaderboard (sorted by test_pr_auc) ===")
    print(leaderboard.to_string(index=False))

    if args.similarity:
        a = preds_val["logreg_balanced"]
        b = preds_val["xgb_baseline"]
        print("\n=== Similarity checks on VAL ===")
        print("Max |diff| (val):", float(np.max(np.abs(a - b))))
        print("Spearman (val):", float(spearmanr(a, b).correlation))
        print(f"Top{args.k} overlap (val):", float(topk_overlap(a, b, args.k)))


if __name__ == "__main__":
    main()
