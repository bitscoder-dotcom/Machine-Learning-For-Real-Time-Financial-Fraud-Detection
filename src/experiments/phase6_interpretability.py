import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_FPR = 0.01


def ensure_dirs():
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/explanations").mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_name: str, csv_path: str):
    df = pd.read_csv(csv_path)

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

    required_cols = [label_col, time_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset/type mismatch. You chose dataset='{dataset_name}', "
            f"but the file is missing required columns: {missing_cols}."
        )

    df = df.sort_values(time_col).reset_index(drop=True)
    return df, label_col, time_col, id_cols_to_drop


def get_feature_columns(df: pd.DataFrame, label_col: str, id_cols_to_drop: list):
    exclude = set([label_col] + id_cols_to_drop)
    feature_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def reduce_ieee_features(train_df: pd.DataFrame, feature_cols: list, max_features: int = 100):
    stats = pd.DataFrame({
        "feature": feature_cols,
        "missing_rate": train_df[feature_cols].isna().mean().values,
        "variance": train_df[feature_cols].var(numeric_only=True).fillna(0).values
    })

    stats = stats.sort_values(
        by=["missing_rate", "variance"],
        ascending=[True, False]
    )

    return stats["feature"].head(max_features).tolist()


def split_into_windows(df: pd.DataFrame, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    n = len(df)
    t1_end = int(n * train_frac)
    t2_end = int(n * (train_frac + val_frac))
    return t1_end, t2_end


def split_train_val_time_ordered(train_df: pd.DataFrame, inner_val_frac=0.20):
    n = len(train_df)
    cut = int(n * (1 - inner_val_frac))
    fit_df = train_df.iloc[:cut]
    val_df = train_df.iloc[cut:]
    return fit_df, val_df


def make_xy(df: pd.DataFrame, feature_cols: list, label_col: str):
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
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid_idx = np.where(fpr <= target_fpr)[0]

    if len(valid_idx) == 0:
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


def fit_fixed_model(train_window_df: pd.DataFrame, feature_cols: list, label_col: str):
    fit_df, val_df = split_train_val_time_ordered(train_window_df, inner_val_frac=0.20)

    X_fit, y_fit = make_xy(fit_df, feature_cols, label_col)
    X_val, y_val = make_xy(val_df, feature_cols, label_col)

    model = build_xgb(y_fit)
    model.fit(X_fit, y_fit)

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold_at_target_fpr(y_val, val_scores, TARGET_FPR)
    val_metrics = evaluate_at_threshold(y_val, val_scores, threshold)

    del X_fit, y_fit, X_val, y_val, val_scores
    gc.collect()

    return model, threshold, val_metrics


def predict_window(model, threshold, df_window, feature_cols, label_col):
    X = df_window.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df_window[label_col].to_numpy(dtype=np.int8, copy=False)
    scores = model.predict_proba(X)[:, 1]
    preds = (scores >= threshold).astype(int)

    out = df_window.copy()
    out["score"] = scores
    out["pred"] = preds
    out["actual"] = y

    metrics = evaluate_at_threshold(y, scores, threshold)

    del X, y, scores, preds
    gc.collect()

    return out, metrics


def sample_for_shap(df_window: pd.DataFrame, max_rows: int, random_state: int = 42):
    if len(df_window) <= max_rows:
        return df_window.copy()
    return df_window.sample(n=max_rows, random_state=random_state).sort_index()


def compute_shap_explanation(model, df_window: pd.DataFrame, feature_cols: list):
    import shap
    X_df = df_window.loc[:, feature_cols].copy()
    X_np = X_df.to_numpy(dtype=np.float32, copy=False)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_np, check_additivity=False)

    return X_df, explanation


def save_global_importance_plot(explanation, feature_cols, dataset_name, window_name, top_n=20):
    import matplotlib.pyplot as plt

    mean_abs_shap = np.abs(explanation.values).mean(axis=0)

    imp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    top_imp = imp.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(top_imp["feature"], top_imp["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"{dataset_name.upper()} {window_name} global SHAP importance")
    plt.tight_layout()

    out_png = f"results/plots/phase6_{dataset_name}_{window_name.lower()}_global_shap.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    out_csv = f"results/metrics/phase6_{dataset_name}_{window_name.lower()}_global_shap.csv"
    imp.to_csv(out_csv, index=False)

    print(f"Saved global SHAP plot: {out_png}")
    print(f"Saved global SHAP table: {out_csv}")

    return imp


def save_local_explanation_plot(explanation, X_df, original_df, idx, dataset_name, case_name, top_n=12):
    import matplotlib.pyplot as plt

    shap_vals = explanation.values[idx]
    row_vals = X_df.iloc[idx].values
    feature_cols = X_df.columns.tolist()

    local_df = pd.DataFrame({
        "feature": feature_cols,
        "feature_value": row_vals,
        "shap_value": shap_vals
    })

    local_df["abs_shap"] = np.abs(local_df["shap_value"])
    local_df = local_df.sort_values("abs_shap", ascending=False).head(top_n).iloc[::-1]

    labels = [
        f"{f}={v:.4g}" if pd.notna(v) else f"{f}=NaN"
        for f, v in zip(local_df["feature"], local_df["feature_value"])
    ]

    plt.figure(figsize=(9, 6))
    plt.barh(labels, local_df["shap_value"])
    plt.xlabel("SHAP value")
    plt.ylabel("Top contributing features")
    plt.title(f"{dataset_name.upper()} local SHAP: {case_name}")
    plt.tight_layout()

    out_png = f"results/plots/phase6_{dataset_name}_{case_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    out_csv = f"results/explanations/phase6_{dataset_name}_{case_name.lower().replace(' ', '_')}.csv"
    local_df.to_csv(out_csv, index=False)

    print(f"Saved local SHAP plot: {out_png}")
    print(f"Saved local SHAP table: {out_csv}")


def get_top_features(importance_df: pd.DataFrame, top_n: int = 10):
    return importance_df.head(top_n)["feature"].tolist()


def compute_stability_summary(t2_imp: pd.DataFrame, t3_imp: pd.DataFrame, dataset_name: str, top_n: int = 10):
    t2_top = get_top_features(t2_imp, top_n)
    t3_top = get_top_features(t3_imp, top_n)

    overlap = sorted(list(set(t2_top).intersection(set(t3_top))))
    union = sorted(list(set(t2_top).union(set(t3_top))))
    jaccard = len(overlap) / len(union) if len(union) > 0 else 0.0

    t2_rank = pd.Series(range(1, len(t2_imp) + 1), index=t2_imp["feature"])
    t3_rank = pd.Series(range(1, len(t3_imp) + 1), index=t3_imp["feature"])
    common_features = t2_imp["feature"].tolist()

    rank_df = pd.DataFrame({
        "t2_rank": t2_rank.loc[common_features].values,
        "t3_rank": t3_rank.loc[common_features].values
    }, index=common_features)

    spearman_like = rank_df["t2_rank"].corr(rank_df["t3_rank"], method="spearman")

    out_txt = f"results/metrics/phase6_{dataset_name}_stability_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Top-{top_n} overlap count: {len(overlap)}\n")
        f.write(f"Top-{top_n} overlap features: {overlap}\n")
        f.write(f"Top-{top_n} Jaccard similarity: {jaccard:.4f}\n")
        f.write(f"Spearman rank correlation across all features: {spearman_like:.4f}\n")

    print(f"Saved stability summary: {out_txt}")

    return {
        "top_n": top_n,
        "overlap_count": len(overlap),
        "overlap_features": overlap,
        "jaccard": jaccard,
        "spearman_rank_corr": spearman_like
    }


def pick_local_cases(pred_df: pd.DataFrame):
    tp_candidates = pred_df[(pred_df["actual"] == 1) & (pred_df["pred"] == 1)].sort_values("score", ascending=False)
    fp_candidates = pred_df[(pred_df["actual"] == 0) & (pred_df["pred"] == 1)].sort_values("score", ascending=False)

    tp_idx = tp_candidates.index[0] if len(tp_candidates) > 0 else None
    fp_idx = fp_candidates.index[0] if len(fp_candidates) > 0 else None

    return tp_idx, fp_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["creditcard", "ieee"])
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--shap_sample", type=int, default=2000)
    args = parser.parse_args()

    ensure_dirs()

    df, label_col, time_col, id_cols_to_drop = load_dataset(args.dataset, args.csv_path)
    feature_cols = get_feature_columns(df, label_col, id_cols_to_drop)

    t1_end, t2_end = split_into_windows(df, 0.70, 0.15, 0.15)

    T1 = df.iloc[:t1_end]
    T2 = df.iloc[t1_end:t2_end]
    T3 = df.iloc[t2_end:]

    if args.dataset == "ieee":
        original_feature_count = len(feature_cols)
        feature_cols = reduce_ieee_features(T1, feature_cols, max_features=args.max_features)
        print(f"Reduced IEEE features from {original_feature_count} to {len(feature_cols)}")

    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(df)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"T1 rows: {len(T1)}")
    print(f"T2 rows: {len(T2)}")
    print(f"T3 rows: {len(T3)}")

    # fixed model from T1
    model, threshold, val_metrics = fit_fixed_model(T1, feature_cols, label_col)
    print("Validation metrics from T1:", val_metrics)
    print("Selected threshold:", threshold)

    # predictions on T2 and T3
    pred_t2, metrics_t2 = predict_window(model, threshold, T2, feature_cols, label_col)
    pred_t3, metrics_t3 = predict_window(model, threshold, T3, feature_cols, label_col)

    print("T2 metrics:", metrics_t2)
    print("T3 metrics:", metrics_t3)

    # SHAP sample windows
    shap_t2_df = sample_for_shap(pred_t2, max_rows=args.shap_sample, random_state=RANDOM_STATE)
    shap_t3_df = sample_for_shap(pred_t3, max_rows=args.shap_sample, random_state=RANDOM_STATE)

    X_t2_df, shap_t2 = compute_shap_explanation(model, shap_t2_df, feature_cols)
    X_t3_df, shap_t3 = compute_shap_explanation(model, shap_t3_df, feature_cols)

    # global importance
    t2_imp = save_global_importance_plot(shap_t2, feature_cols, args.dataset, "T2", top_n=20)
    t3_imp = save_global_importance_plot(shap_t3, feature_cols, args.dataset, "T3", top_n=20)

    # local cases from T3 first
    tp_idx_t3, fp_idx_t3 = pick_local_cases(shap_t3_df)

    if tp_idx_t3 is not None:
        pos_t3 = shap_t3_df.index.get_loc(tp_idx_t3)
        save_local_explanation_plot(shap_t3, X_t3_df, shap_t3_df, pos_t3, args.dataset, "correct_flagged_fraud_t3", top_n=12)
    else:
        print("No true positive found in sampled T3 for local explanation.")

    if fp_idx_t3 is not None:
        pos_fp_t3 = shap_t3_df.index.get_loc(fp_idx_t3)
        save_local_explanation_plot(shap_t3, X_t3_df, shap_t3_df, pos_fp_t3, args.dataset, "false_positive_t3", top_n=12)
    else:
        print("No false positive found in sampled T3 for local explanation.")

    # stability summary
    stability = compute_stability_summary(t2_imp, t3_imp, args.dataset, top_n=10)
    print("Stability summary:", stability)

    # save high-level run summary
    summary_df = pd.DataFrame([
        {"window": "T2", **metrics_t2},
        {"window": "T3", **metrics_t3},
    ])
    summary_path = f"results/metrics/phase6_{args.dataset}_window_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved window metrics: {summary_path}")


if __name__ == "__main__":
    main()