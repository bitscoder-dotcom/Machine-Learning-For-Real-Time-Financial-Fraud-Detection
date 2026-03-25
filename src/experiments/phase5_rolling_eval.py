import os
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_FPR = 0.01


def ensure_dirs():
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_name: str, csv_path: str):
    """
    Loads dataset and returns:
    df, label_col, time_col, id_cols_to_drop
    """
    df = pd.read_csv(csv_path)

    # reduce memory immediately
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    if dataset_name == "creditcard":
        label_col = "Class"
        time_col = "Time"
        id_cols_to_drop = []

    elif dataset_name == "ieee":
        label_col = "isFraud"
        time_col = "TransactionDT"
        id_cols_to_drop = ["TransactionID"] if "TransactionID" in df.columns else []

    else:
        raise ValueError("dataset_name must be 'creditcard' or 'ieee'")

    df = df.sort_values(time_col).reset_index(drop=True)
    return df, label_col, time_col, id_cols_to_drop


def get_feature_columns(df: pd.DataFrame, label_col: str, time_col: str, id_cols_to_drop: list):
    """
    Keep numeric columns only, and exclude label/IDs.
    Keep time_col as a feature because your earlier phases used numeric baseline features.
    """
    exclude = set([label_col] + id_cols_to_drop)
    feature_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols

def reduce_ieee_features(train_df: pd.DataFrame, feature_cols: list, max_features: int = 100):
    """
    Memory-safe, train-window-only feature reduction for IEEE-CIS.

    We rank features using:
    1. lower missing rate first
    2. higher variance second

    This uses only the current training window, so it stays leakage-safe.
    """
    stats = pd.DataFrame({
        "feature": feature_cols,
        "missing_rate": train_df[feature_cols].isna().mean().values,
        "variance": train_df[feature_cols].var(numeric_only=True).fillna(0).values
    })

    stats = stats.sort_values(
        by=["missing_rate", "variance"],
        ascending=[True, False]
    )

    selected = stats["feature"].head(max_features).tolist()
    return selected


def split_into_windows(df: pd.DataFrame, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """
    Return boundary indices only.
    This avoids making huge DataFrame copies.
    """
    n = len(df)
    t1_end = int(n * train_frac)
    t2_end = int(n * (train_frac + val_frac))
    return t1_end, t2_end


def split_train_val_time_ordered(train_df: pd.DataFrame, inner_val_frac=0.20):
    """
    No deep copies here.
    """
    n = len(train_df)
    cut = int(n * (1 - inner_val_frac))
    fit_df = train_df.iloc[:cut]
    val_df = train_df.iloc[cut:]
    return fit_df, val_df


def make_xy(df: pd.DataFrame, feature_cols: list, label_col: str):
    """
    Force float32 for features and int8 for labels to reduce memory.
    """
    X = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[label_col].to_numpy(dtype=np.int8, copy=False)
    return X, y


def build_xgb(y_train: np.ndarray):
    fraud_count = np.sum(y_train == 1)
    nonfraud_count = np.sum(y_train == 0)

    if fraud_count == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = nonfraud_count / fraud_count

    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.5,
        colsample_bytree=0.5,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        max_bin=64
    )
    return model


def choose_threshold_at_target_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = TARGET_FPR):
    """
    Pick the threshold from validation scores such that FPR <= target_fpr,
    while maximizing TPR among valid thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
        # fallback: very strict threshold
        return np.max(scores) + 1e-12

    best_local = valid_idx[np.argmax(tpr[valid_idx])]
    return thresholds[best_local]


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    pr_auc = average_precision_score(y_true, scores)

    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "pr_auc": float(pr_auc),
        "recall_at_1pct_fpr": float(recall),
        "actual_fpr": float(actual_fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def fit_and_prepare(train_window_df: pd.DataFrame, feature_cols: list, label_col: str):
    fit_df, val_df = split_train_val_time_ordered(train_window_df, inner_val_frac=0.20)

    X_fit, y_fit = make_xy(fit_df, feature_cols, label_col)
    X_val, y_val = make_xy(val_df, feature_cols, label_col)

    model = build_xgb(y_fit)
    model.fit(X_fit, y_fit)

    # free fit arrays once model is trained
    del X_fit, y_fit
    gc.collect()

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold_at_target_fpr(y_val, val_scores, TARGET_FPR)

    val_metrics = evaluate_at_threshold(y_val, val_scores, threshold)

    del X_val, y_val, val_scores

    return model, threshold, val_metrics


def test_model(model, threshold, test_df: pd.DataFrame, feature_cols: list, label_col: str):
    X_test, y_test = make_xy(test_df, feature_cols, label_col)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)

    del X_test, y_test, test_scores
    gc.collect()

    return test_metrics


def run_fixed_strategy(T1, T2, T3, feature_cols, label_col):
    """
    Fixed model:
      train on T1 once
      choose threshold on T1 validation slice once
      test on T2 and T3 with same model + same threshold
    """
    model, threshold, val_metrics = fit_and_prepare(T1, feature_cols, label_col)

    t2_metrics = test_model(model, threshold, T2, feature_cols, label_col)
    t3_metrics = test_model(model, threshold, T3, feature_cols, label_col)

    rows = [
        {
            "strategy": "fixed",
            "train_window": "T1",
            "test_window": "T2",
            "threshold": threshold,
            **t2_metrics
        },
        {
            "strategy": "fixed",
            "train_window": "T1",
            "test_window": "T3",
            "threshold": threshold,
            **t3_metrics
        }
    ]

    return rows, val_metrics


def run_periodic_retraining_strategy(T1, T2, T3, feature_cols, label_col):
    """
    Periodic retraining:
      train on T1 -> test on T2
      retrain on T1+T2 -> test on T3
    """
    rows = []

    # Step A: T1 -> T2
    model_1, threshold_1, val_metrics_1 = fit_and_prepare(T1, feature_cols, label_col)
    t2_metrics = test_model(model_1, threshold_1, T2, feature_cols, label_col)

    rows.append({
        "strategy": "periodic_retraining",
        "train_window": "T1",
        "test_window": "T2",
        "threshold": threshold_1,
        **t2_metrics
    })

    # Step B: T1+T2 -> T3
    T1_T2 = pd.concat([T1, T2], axis=0, ignore_index=True)
    model_2, threshold_2, val_metrics_2 = fit_and_prepare(T1_T2, feature_cols, label_col)
    t3_metrics = test_model(model_2, threshold_2, T3, feature_cols, label_col)

    rows.append({
        "strategy": "periodic_retraining",
        "train_window": "T1+T2",
        "test_window": "T3",
        "threshold": threshold_2,
        **t3_metrics
    })

    return rows, val_metrics_1, val_metrics_2


def save_results_csv(results_df: pd.DataFrame, dataset_name: str):
    out_path = f"results/metrics/phase5_{dataset_name}_rolling_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")


def plot_metric(results_df: pd.DataFrame, dataset_name: str, metric_col: str, ylabel: str, filename_suffix: str):
    plt.figure(figsize=(7, 4))

    for strategy in results_df["strategy"].unique():
        d = results_df[results_df["strategy"] == strategy].copy()

        x_vals = d["test_window"].tolist()
        y_vals = d[metric_col].tolist()

        plt.plot(x_vals, y_vals, marker="o", label=strategy)

    plt.xlabel("Test window")
    plt.ylabel(ylabel)
    plt.title(f"{dataset_name.upper()} - {ylabel} over time")
    plt.legend()
    plt.tight_layout()

    out_path = f"results/plots/phase5_{dataset_name}_{filename_suffix}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["creditcard", "ieee"])
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--max_features", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()

    df, label_col, time_col, id_cols_to_drop = load_dataset(args.dataset, args.csv_path)

    feature_cols = get_feature_columns(df, label_col, time_col, id_cols_to_drop)

    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(df)}")
    print(f"Label column: {label_col}")
    print(f"Time column: {time_col}")
    print(f"Number of features: {len(feature_cols)}")

    t1_end, t2_end = split_into_windows(df, 0.70, 0.15, 0.15)

    T1 = df.iloc[:t1_end]
    T2 = df.iloc[t1_end:t2_end]
    T3 = df.iloc[t2_end:]

    if args.dataset == "ieee":
        original_feature_count = len(feature_cols)
        feature_cols = reduce_ieee_features(T1, feature_cols, max_features=args.max_features)
        print(f"Reduced IEEE features from {original_feature_count} to {len(feature_cols)}")

    print(f"T1 rows: {len(T1)}")
    print(f"T2 rows: {len(T2)}")
    print(f"T3 rows: {len(T3)}")

    fixed_rows, fixed_val_metrics = run_fixed_strategy(T1, T2, T3, feature_cols, label_col)
    retrain_rows, retrain_val_metrics_1, retrain_val_metrics_2 = run_periodic_retraining_strategy(
        T1, T2, T3, feature_cols, label_col
    )

    results_df = pd.DataFrame(fixed_rows + retrain_rows)

    # Keep a clean order
    results_df = results_df[
        [
            "strategy",
            "train_window",
            "test_window",
            "threshold",
            "pr_auc",
            "recall_at_1pct_fpr",
            "actual_fpr",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
    ]

    save_results_csv(results_df, args.dataset)

    plot_metric(
        results_df=results_df,
        dataset_name=args.dataset,
        metric_col="pr_auc",
        ylabel="PR-AUC",
        filename_suffix="pr_auc_over_time"
    )

    plot_metric(
        results_df=results_df,
        dataset_name=args.dataset,
        metric_col="recall_at_1pct_fpr",
        ylabel="Recall at validation-selected 1% FPR threshold",
        filename_suffix="recall_at_1pct_fpr_over_time"
    )

    print("\nValidation metrics used internally for threshold selection:")
    print("Fixed strategy (from T1 validation):", fixed_val_metrics)
    print("Periodic retraining step 1 (from T1 validation):", retrain_val_metrics_1)
    print("Periodic retraining step 2 (from T1+T2 validation):", retrain_val_metrics_2)

    print("\nFinal Phase 5 results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()