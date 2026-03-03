import os
import numpy as np
import pandas as pd

from src.utils.splits import baseline_split, time_aware_split
from src.utils.leakage_checks import warn_suspicious_columns, assert_no_overlap_ids, assert_time_order

def dataset_card(df, name, label_col, time_col=None, out_md=None):
    lines = []
    lines.append(f"# Dataset card: {name}\n")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns: {df.shape[1]:,}")
    lines.append(f"- Label: `{label_col}` (pos rate={df[label_col].mean():.6f})")
    if time_col and time_col in df.columns:
        lines.append(f"- Time/order field: `{time_col}` (min={df[time_col].min()}, max={df[time_col].max()})")

    miss = (df.isna().mean().sort_values(ascending=False) * 100)
    top_miss = miss[miss > 0].head(20)
    lines.append("\n## Missing values (top 20)\n")
    if len(top_miss) == 0:
        lines.append("None.\n")
    else:
        lines.append(top_miss.to_string())

    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return "\n".join(lines)

def save_split_npz(out_path, idx_train, idx_val, idx_test, meta: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        train_idx=idx_train.astype(int),
                        val_idx=idx_val.astype(int),
                        test_idx=idx_test.astype(int),
                        meta=meta)

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw = os.path.join(root, "data", "raw")
    out_cards = os.path.join(root, "results", "data_cards")
    out_splits = os.path.join(root, "results", "splits")

    datasets = [
        # name, filename, label, time_col, id_cols(optional), group_cols(optional)
        ("creditcard", "creditcard.csv", "Class", "Time", None, None),
        ("ieee_cis", "train_transaction.csv", "isFraud", "TransactionDT", "TransactionID", None),
    ]

    for name, fname, label, time_col, id_col, group_cols in datasets:
        path = os.path.join(raw, fname)
        if not os.path.exists(path):
            print(f"Skip (missing): {path}")
            continue

        df = pd.read_csv(path)
        warn_suspicious_columns(df)

        # dataset card
        dataset_card(df, name, label, time_col=time_col,
                     out_md=os.path.join(out_cards, f"{name}_data_card.md"))

        # baseline split
        tr, va, te = baseline_split(df, label, seed=42, frac=(0.70,0.15,0.15), group_cols=group_cols)
        if id_col and id_col in df.columns:
            assert_no_overlap_ids(df, tr, va, te, id_col)

        save_split_npz(
            os.path.join(out_splits, f"{name}_baseline_seed42_70_15_15.npz"),
            tr, va, te,
            meta={"method":"baseline", "seed":42, "label":label, "time_col":time_col, "group_cols":group_cols}
        )

        # time-aware split
        if time_col in df.columns:
            tr, va, te = time_aware_split(df, label, time_col, frac=(0.70,0.15,0.15), group_cols=group_cols)
            assert_time_order(df, tr, va, te, time_col)
            if id_col and id_col in df.columns:
                assert_no_overlap_ids(df, tr, va, te, id_col)

            save_split_npz(
                os.path.join(out_splits, f"{name}_time_{time_col}_70_15_15.npz"),
                tr, va, te,
                meta={"method":"time_aware", "label":label, "time_col":time_col, "group_cols":group_cols}
            )

        print(f"Phase 2 prepared for {name}")

if __name__ == "__main__":
    main()