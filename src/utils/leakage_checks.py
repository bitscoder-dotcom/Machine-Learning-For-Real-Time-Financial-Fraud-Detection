import numpy as np

LEAKY_HINTS = [
    "label", "target", "fraud", "isfraud", "class",
    "outcome", "future", "next", "post", "confirmed"
]

def warn_suspicious_columns(df):
    cols = [c for c in df.columns if any(h in c.lower() for h in LEAKY_HINTS)]
    if cols:
        print("Suspicious/leak-prone columns (review & usually drop from features):", cols)

def assert_no_overlap_ids(df, idx_train, idx_val, idx_test, id_cols):
    def ids(ix):
        if isinstance(id_cols, str):
            return set(df.loc[ix, id_cols].astype(str).tolist())
        # composite id
        return set(df.loc[ix, id_cols].astype(str).agg("||".join, axis=1).tolist())

    a, b, c = ids(idx_train), ids(idx_val), ids(idx_test)
    assert a.isdisjoint(b), "ID overlap: train vs val"
    assert a.isdisjoint(c), "ID overlap: train vs test"
    assert b.isdisjoint(c), "ID overlap: val vs test"

def assert_time_order(df, idx_train, idx_val, idx_test, time_col):
    tr_max = df.loc[idx_train, time_col].max()
    va_min = df.loc[idx_val, time_col].min()
    va_max = df.loc[idx_val, time_col].max()
    te_min = df.loc[idx_test, time_col].min()
    assert tr_max <= va_min, f"Time leakage: train max ({tr_max}) > val min ({va_min})"
    assert va_max <= te_min, f"Time leakage: val max ({va_max}) > test min ({te_min})"