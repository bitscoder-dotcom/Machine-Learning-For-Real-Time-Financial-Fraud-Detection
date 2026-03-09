import argparse
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd

from src.utils.split_manager import project_root


def parse_ts(x):
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x))
    except Exception:
        return None


def norm_path(p: str) -> str:
    return str(p).replace("\\", "/") if p else ""


def expected_split_file(root: Path, cfg: dict) -> str:
    train_on = cfg.get("train_on_split", "baseline")
    if train_on == "time":
        p = root / cfg.get("split_time_path", "")
    else:
        p = root / cfg.get("split_baseline_path", cfg.get("split_path", ""))
    return norm_path(str(p.resolve()))


def get_metric(d: dict, prefix: str, name: str, k: int):
    """
    Handles your mixed naming across scripts:
    - *_precision_at_k vs *_precision_at_1000
    - *_fraud_found_at_k vs *_fraud_found_at_1000
    """
    # direct
    key = f"{prefix}_{name}"
    if key in d:
        return d[key]

    # k variants
    if name in ("precision_at_k", "fraud_found_at_k"):
        key1 = f"{prefix}_{name}"                       # e.g. test_precision_at_k
        key2 = f"{prefix}_{name.replace('_at_k', f'_at_{k}')}"  # e.g. test_precision_at_1000
        key3 = f"{prefix}_{name.replace('_at_k', '_at_1000')}"  # fallback
        for kk in (key1, key2, key3):
            if kk in d:
                return d[kk]

    return None


def load_latest_metrics(root: Path, cfg: dict, wanted_models: list[str], k: int):
    metrics_dir = root / "results" / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing metrics dir: {metrics_dir}")

    want_split = expected_split_file(root, cfg)

    rows = []
    for p in sorted(metrics_dir.glob("*.yaml")):
        data = yaml.safe_load(open(p, "r", encoding="utf-8"))
        if not isinstance(data, dict):
            continue

        model = data.get("model")
        if model not in wanted_models:
            continue

        # Filter to the current config’s split file (time vs baseline, dataset-specific)
        sf = norm_path(str(data.get("split_file", "")))
        if want_split and sf and sf != want_split:
            continue

        ts = parse_ts(data.get("timestamp")) or datetime.fromtimestamp(p.stat().st_mtime)

        rows.append({
            "timestamp": ts,
            "model": model,
            "split_used": data.get("split_used"),
            "split_file": data.get("split_file"),
            "k": data.get("k", k),
            "target_fpr": data.get("target_fpr"),
            "op_threshold": data.get("op_threshold"),
            "op_val_fpr": data.get("op_val_fpr"),
            "op_val_recall": data.get("op_val_recall"),

            "train_pr_auc": data.get("train_pr_auc"),
            "val_pr_auc": data.get("val_pr_auc"),
            "test_pr_auc": data.get("test_pr_auc"),

            "train_recall_at_fpr": data.get("train_recall_at_fpr"),
            "val_recall_at_fpr": data.get("val_recall_at_fpr"),
            "test_recall_at_fpr": data.get("test_recall_at_fpr"),

            "train_precision@k": get_metric(data, "train", "precision_at_k", k),
            "val_precision@k": get_metric(data, "val", "precision_at_k", k),
            "test_precision@k": get_metric(data, "test", "precision_at_k", k),

            "train_fraud_found@k": get_metric(data, "train", "fraud_found_at_k", k),
            "val_fraud_found@k": get_metric(data, "val", "fraud_found_at_k", k),
            "test_fraud_found@k": get_metric(data, "test", "fraud_found_at_k", k),

            "test_ms_per_row": data.get("test_ms_per_row"),
            "test_rows_per_s": data.get("test_rows_per_s"),

            "model_path": data.get("model_path"),
            "metrics_path": str(p.relative_to(root)),
        })

    if not rows:
        raise FileNotFoundError(
            "No matching metrics YAMLs found for these models + config split. "
            "Check split_file in your metrics and the config you passed."
        )

    df = pd.DataFrame(rows)

    # pick latest run per model
    df = df.sort_values("timestamp").groupby("model", as_index=False).tail(1)

    # nice formatting
    df["run_timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop(columns=["timestamp"])

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--models", default="logreg_balanced,rf_baseline,xgb_baseline",
                    help="Comma-separated model names as stored in metrics YAML")
    ap.add_argument("--sort", default="test_pr_auc",
                    help="Column to sort by (default: test_pr_auc)")
    args = ap.parse_args()

    root = project_root()
    cfg = yaml.safe_load(open(root / args.config, "r", encoding="utf-8"))
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    k = int(cfg.get("k", args.k))

    df = load_latest_metrics(root, cfg, wanted, k)

    sort_col = args.sort
    if sort_col not in df.columns:
        raise ValueError(f"Unknown sort column '{sort_col}'. Available: {list(df.columns)}")

    leaderboard = df.sort_values(sort_col, ascending=False)

    # keep the table tight
    cols = [
        "model", "run_timestamp", "split_used", "target_fpr",
        "val_pr_auc", "test_pr_auc",
        "val_recall_at_fpr", "test_recall_at_fpr",
        "val_precision@k", "test_precision@k",
        "val_fraud_found@k", "test_fraud_found@k",
        "op_threshold", "op_val_fpr", "op_val_recall",
        "test_ms_per_row", "test_rows_per_s",
        "model_path", "metrics_path"
    ]
    cols = [c for c in cols if c in leaderboard.columns]

    pd.set_option("display.max_colwidth", 120)
    print("\n=== Phase 3 Leaderboard (latest run per model) ===")
    print(f"Config: {args.config}")
    print(leaderboard[cols].to_string(index=False))


if __name__ == "__main__":
    main()