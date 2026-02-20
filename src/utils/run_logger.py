import csv
import os
from typing import Dict, Any, List
import numpy as np

# 1) Fixed schema (ordered columns). Add to this list when you introduce new metrics.
RUN_SCHEMA: List[str] = [
    "timestamp",
    "model",
    "split",
    "seed",

    "n_train", "n_val", "n_test",
    "fraud_rate_train", "fraud_rate_val", "fraud_rate_test",
    "fraud_count_train", "fraud_count_val", "fraud_count_test",

    # model-specific (will be blank for models that don't have it)
    "scale_pos_weight",

    # train metrics
    "train_pr_auc",
    "train_recall_at_precision_0.10",
    "train_recall_at_precision_0.20",
    "train_precision_at_1000",
    "train_fraud_found_at_1000",

    # val metrics
    "val_pr_auc",
    "val_recall_at_precision_0.10",
    "val_recall_at_precision_0.20",
    "val_precision_at_1000",
    "val_fraud_found_at_1000",

    # test metrics
    "test_pr_auc",
    "test_recall_at_precision_0.10",
    "test_recall_at_precision_0.20",
    "test_precision_at_1000",
    "test_fraud_found_at_1000",

    # traceability
    "model_path",
    "metrics_path",
]

def _to_builtin(v: Any) -> Any:
    """Convert numpy scalars to python types so CSV writes cleanly."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def append_run_csv(row: Dict[str, Any], path: str = "results/metrics/runs.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Ensure schema coverage: missing keys -> empty
    clean_row = {k: _to_builtin(row.get(k, "")) for k in RUN_SCHEMA}

    file_exists = os.path.exists(path)

    # If file exists, ensure header matches schema (avoid silent corruption)
    if file_exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header and existing_header != RUN_SCHEMA:
            raise ValueError(
                "runs.csv header does not match RUN_SCHEMA. "
                "Either update RUN_SCHEMA to match or recreate runs.csv."
            )

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_SCHEMA)
        if not file_exists:
            writer.writeheader()
        writer.writerow(clean_row)
