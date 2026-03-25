import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--split-path", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subtype-col", default=None)
    return p.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_indices(npz_path):
    arrs = np.load(npz_path, allow_pickle=True)
    key_sets = [
        ("train_idx", "val_idx", "test_idx"),
        ("train_indices", "val_indices", "test_indices"),
        ("train", "val", "test"),
    ]
    for keys in key_sets:
        if all(k in arrs for k in keys):
            return arrs[keys[0]], arrs[keys[1]], arrs[keys[2]]
    raise KeyError(f"Could not find train/val/test indices in {npz_path}. Keys found: {list(arrs.keys())}")

def get_required_columns(cfg, model, subtype_col=None):
    required = set()

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        required.update(list(model_features))

    required.add(cfg["target_col"])

    time_col = cfg.get("time_col")
    if time_col:
        required.add(time_col)

    # keep these for error analysis if present
    required.add("Amount")
    required.add("TransactionAmt")

    if subtype_col:
        required.add(subtype_col)

    # only keep columns that actually exist in the file
    header = pd.read_csv(cfg["dataset_path"], nrows=0)
    available = set(header.columns)
    return [c for c in required if c in available]


def load_subset_rows(csv_path, val_idx, test_idx, usecols, chunksize=50000):
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    val_parts = []
    test_parts = []
    start = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        n = len(chunk)
        global_idx = np.arange(start, start + n)

        val_mask = np.isin(global_idx, val_idx)
        test_mask = np.isin(global_idx, test_idx)

        if val_mask.any():
            val_parts.append(chunk.loc[val_mask].copy())

        if test_mask.any():
            test_parts.append(chunk.loc[test_mask].copy())

        start += n

    raw_val = pd.concat(val_parts, ignore_index=True)
    raw_test = pd.concat(test_parts, ignore_index=True)
    return raw_val, raw_test

def build_feature_matrix(df, label_col, drop_cols=None, subtype_col=None):
    drop_cols = drop_cols or []
    cols_to_drop = [label_col] + list(drop_cols)

    if subtype_col and subtype_col in df.columns:
        cols_to_drop.append(subtype_col)

    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")

    # stay aligned with your earlier baseline pipeline: numeric only
    X = X.select_dtypes(include=[np.number])
    y = df[label_col].astype(int).copy()
    return X, y


def align_features_for_model(X, model):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing features expected by model: {missing[:10]}")
        return X.loc[:, feature_names]
    return X


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Model does not support predict_proba or decision_function")


def choose_threshold_at_fpr(y_true, probs, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 1.0
    best = valid[np.argmax(tpr[valid])]
    return float(thresholds[best])


def precision_at_k(y_true, probs, k=1000):
    k = min(k, len(y_true))
    order = np.argsort(-probs)[:k]
    y_top = np.asarray(y_true)[order]
    return float(np.mean(y_top))


def fraud_found_at_k(y_true, probs, k=1000):
    k = min(k, len(y_true))
    order = np.argsort(-probs)[:k]
    y_top = np.asarray(y_true)[order]
    return int(np.sum(y_top))


def recall_from_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return float(tp / positives)


def actual_fpr_from_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    negatives = np.sum(y_true == 0)
    if negatives == 0:
        return 0.0
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fp / negatives)


def bootstrap_metric_cis(y_true, probs, threshold, k=1000, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = np.asarray(y_true)[idx]
        p_b = np.asarray(probs)[idx]

        row = {
            "pr_auc": average_precision_score(y_b, p_b) if len(np.unique(y_b)) > 1 else np.nan,
            "recall_at_operating_threshold": recall_from_threshold(y_b, p_b, threshold),
            "precision_at_1000": precision_at_k(y_b, p_b, k=k),
            "actual_fpr_at_operating_threshold": actual_fpr_from_threshold(y_b, p_b, threshold),
            "brier_score": brier_score_loss(y_b, p_b),
        }
        rows.append(row)

    boot = pd.DataFrame(rows)
    summary_rows = []
    for col in boot.columns:
        s = boot[col].dropna()
        summary_rows.append(
            {
                "metric": col,
                "mean": float(s.mean()),
                "ci_lower_95": float(s.quantile(0.025)),
                "ci_upper_95": float(s.quantile(0.975)),
            }
        )
    return boot, pd.DataFrame(summary_rows)


def make_reliability_table(y_true, probs, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "score": probs}).copy()
    df["bin"] = pd.qcut(df["score"], q=min(n_bins, df["score"].nunique()), duplicates="drop")
    out = (
        df.groupby("bin", observed=False)
        .agg(
            count=("y_true", "size"),
            observed_rate=("y_true", "mean"),
            mean_predicted_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )
    return out


def plot_reliability_curve(y_true, probs, out_path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted fraud probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Reliability curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def add_common_columns(df, time_col=None):
    df = df.copy()

    if time_col and time_col in df.columns:
        n_bins = min(10, df[time_col].nunique())
        if n_bins > 1:
            df["time_bin"] = pd.qcut(df[time_col], q=n_bins, duplicates="drop")

    if "Amount" in df.columns:
        n_bins = min(10, df["Amount"].nunique())
        if n_bins > 1:
            df["amount_bin"] = pd.qcut(df["Amount"], q=n_bins, duplicates="drop")

    if "TransactionAmt" in df.columns:
        n_bins = min(10, df["TransactionAmt"].nunique())
        if n_bins > 1:
            df["transaction_amt_bin"] = pd.qcut(df["TransactionAmt"], q=n_bins, duplicates="drop")

    n_bins = min(10, df["score"].nunique())
    if n_bins > 1:
        df["score_bin"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")

    return df


def summarize_error_clusters(df, base_mask, error_mask, group_col):
    if group_col not in df.columns:
        return None

    base = df.loc[base_mask].groupby(group_col, observed=False).size().rename("base_count")
    err = df.loc[error_mask].groupby(group_col, observed=False).size().rename("error_count")

    out = pd.concat([base, err], axis=1).fillna(0).reset_index()
    out["base_count"] = out["base_count"].astype(int)
    out["error_count"] = out["error_count"].astype(int)
    out["error_rate_within_bin"] = np.where(
        out["base_count"] > 0, out["error_count"] / out["base_count"], np.nan
    )
    return out.sort_values("error_rate_within_bin", ascending=False)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    label_col = cfg["target_col"]
    drop_cols = list(cfg.get("drop_cols", []))
    id_col = cfg.get("id_col")
    time_col = cfg.get("time_col")

    if id_col:
        drop_cols.append(id_col)

    out_metrics = Path("results/metrics")
    out_plots = Path("results/plots")
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    _, val_idx, test_idx = resolve_split_indices(args.split_path)
    print("VAL SIZE:", len(val_idx))
    print("TEST SIZE:", len(test_idx))
    print("VAL/TEST OVERLAP:", len(np.intersect1d(val_idx, test_idx)))
    print("VAL HEAD:", val_idx[:5])
    print("TEST HEAD:", test_idx[:5])
    model = joblib.load(args.model_path)

    print("MODEL TYPE:", type(model))
    print("HAS feature_names_in_:", hasattr(model, "feature_names_in_"))
    if hasattr(model, "feature_names_in_"):
        print("N FEATURES EXPECTED:", len(model.feature_names_in_))
        print("FIRST 20 FEATURES:", list(model.feature_names_in_)[:20])

    required_cols = get_required_columns(cfg, model, subtype_col=args.subtype_col)

    required_cols = get_required_columns(cfg, model, subtype_col=args.subtype_col)

    raw_val, raw_test = load_subset_rows(
        csv_path=cfg["dataset_path"],
        val_idx=val_idx,
        test_idx=test_idx,
        usecols=required_cols,
        chunksize=50000,
    )

    X_val, y_val = build_feature_matrix(
        raw_val,
        label_col=label_col,
        drop_cols=drop_cols,
        subtype_col=args.subtype_col,
    )
    X_test, y_test = build_feature_matrix(
        raw_test,
        label_col=label_col,
        drop_cols=drop_cols,
        subtype_col=args.subtype_col,
    )

    X_val = align_features_for_model(X_val, model)
    X_test = align_features_for_model(X_test, model)

    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("X_val columns first 20:", list(X_val.columns)[:20])
    print("X_test columns first 20:", list(X_test.columns)[:20])

    val_probs = get_probabilities(model, X_val)
    test_probs = get_probabilities(model, X_test)

    val_probs = get_probabilities(model, X_val)
    test_probs = get_probabilities(model, X_test)

    threshold = choose_threshold_at_fpr(y_val, val_probs, target_fpr=args.target_fpr)
    test_pred = (test_probs >= threshold).astype(int)

    point_metrics = {
        "dataset": args.dataset_name,
        "model_path": args.model_path,
        "split_path": args.split_path,
        "target_fpr_on_validation": args.target_fpr,
        "operating_threshold": float(threshold),
        "test_pr_auc": float(average_precision_score(y_test, test_probs)),
        "test_brier_score": float(brier_score_loss(y_test, test_probs)),
        "test_recall_at_operating_threshold": float(recall_from_threshold(y_test, test_probs, threshold)),
        "test_actual_fpr_at_operating_threshold": float(actual_fpr_from_threshold(y_test, test_probs, threshold)),
        "test_precision_at_1000": float(precision_at_k(y_test, test_probs, 1000)),
        "test_fraud_found_at_1000": int(fraud_found_at_k(y_test, test_probs, 1000)),
        "n_test": int(len(y_test)),
        "n_test_fraud": int(np.sum(y_test == 1)),
    }

    reliability_png = out_plots / f"phase8_{args.dataset_name}_reliability_curve.png"
    plot_reliability_curve(y_test, test_probs, reliability_png, n_bins=args.n_bins)

    reliability_table = make_reliability_table(y_test, test_probs, n_bins=args.n_bins)
    reliability_table.to_csv(
        out_metrics / f"phase8_{args.dataset_name}_reliability_table.csv",
        index=False,
    )

    _, ci_summary = bootstrap_metric_cis(
        y_true=y_test.to_numpy(),
        probs=np.asarray(test_probs),
        threshold=threshold,
        k=1000,
        n_boot=args.bootstrap_iters,
        seed=args.seed,
    )
    ci_summary.to_csv(
        out_metrics / f"phase8_{args.dataset_name}_bootstrap_ci.csv",
        index=False,
    )

    with open(out_metrics / f"phase8_{args.dataset_name}_point_metrics.json", "w", encoding="utf-8") as f:
        json.dump(point_metrics, f, indent=2)

    analysis_cols = []

    if time_col and time_col in raw_test.columns:
        analysis_cols.append(time_col)

    for c in ["Amount", "TransactionAmt"]:
        if c in raw_test.columns:
            analysis_cols.append(c)

    if args.subtype_col and args.subtype_col in raw_test.columns:
        analysis_cols.append(args.subtype_col)

    analysis_df = raw_test.loc[:, analysis_cols].copy()

    analysis_df["y_true"] = y_test.to_numpy()
    analysis_df["score"] = np.asarray(test_probs)
    analysis_df["y_pred"] = test_pred
    analysis_df["error_type"] = "correct"
    analysis_df.loc[(analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 1), "error_type"] = "false_positive"
    analysis_df.loc[(analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 0), "error_type"] = "false_negative"
    analysis_df = add_common_columns(analysis_df, time_col=time_col)

    fp_df = analysis_df.loc[analysis_df["error_type"] == "false_positive"].sort_values("score", ascending=False)
    fn_df = analysis_df.loc[analysis_df["error_type"] == "false_negative"].sort_values("score", ascending=False)

    fp_df.head(200).to_csv(out_metrics / f"phase8_{args.dataset_name}_top_false_positives.csv", index=False)
    fn_df.head(200).to_csv(out_metrics / f"phase8_{args.dataset_name}_top_false_negatives.csv", index=False)

    fp_base_mask = analysis_df["y_true"] == 0
    fp_error_mask = analysis_df["error_type"] == "false_positive"
    fn_base_mask = analysis_df["y_true"] == 1
    fn_error_mask = analysis_df["error_type"] == "false_negative"

    cluster_tables = []
    for col in ["score_bin", "time_bin", "amount_bin", "transaction_amt_bin"]:
        fp_tab = summarize_error_clusters(analysis_df, fp_base_mask, fp_error_mask, col)
        if fp_tab is not None:
            fp_tab.insert(0, "analysis", "false_positive")
            fp_tab.insert(1, "group_col", col)
            cluster_tables.append(fp_tab)

        fn_tab = summarize_error_clusters(analysis_df, fn_base_mask, fn_error_mask, col)
        if fn_tab is not None:
            fn_tab.insert(0, "analysis", "false_negative")
            fn_tab.insert(1, "group_col", col)
            cluster_tables.append(fn_tab)

    if args.subtype_col and args.subtype_col in analysis_df.columns:
        fp_tab = summarize_error_clusters(analysis_df, fp_base_mask, fp_error_mask, args.subtype_col)
        if fp_tab is not None:
            fp_tab.insert(0, "analysis", "false_positive")
            fp_tab.insert(1, "group_col", args.subtype_col)
            cluster_tables.append(fp_tab)

        fn_tab = summarize_error_clusters(analysis_df, fn_base_mask, fn_error_mask, args.subtype_col)
        if fn_tab is not None:
            fn_tab.insert(0, "analysis", "false_negative")
            fn_tab.insert(1, "group_col", args.subtype_col)
            cluster_tables.append(fn_tab)

    if cluster_tables:
        error_clusters = pd.concat(cluster_tables, ignore_index=True)
        error_clusters.to_csv(out_metrics / f"phase8_{args.dataset_name}_error_clusters.csv", index=False)

    with open(out_metrics / f"phase8_{args.dataset_name}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Target FPR on validation: {args.target_fpr}\n")
        f.write(f"Operating threshold: {threshold:.6f}\n")
        f.write(f"Test PR-AUC: {point_metrics['test_pr_auc']:.6f}\n")
        f.write(f"Test Brier score: {point_metrics['test_brier_score']:.6f}\n")
        f.write(f"Test Recall at operating threshold: {point_metrics['test_recall_at_operating_threshold']:.6f}\n")
        f.write(f"Test actual FPR at operating threshold: {point_metrics['test_actual_fpr_at_operating_threshold']:.6f}\n")
        f.write(f"Test Precision@1000: {point_metrics['test_precision_at_1000']:.6f}\n")
        f.write(f"Test FraudFound@1000: {point_metrics['test_fraud_found_at_1000']}\n")
        f.write(f"Reliability plot: {reliability_png}\n")

    print("Phase 8 analysis complete.")
    print(json.dumps(point_metrics, indent=2))


if __name__ == "__main__":
    main()