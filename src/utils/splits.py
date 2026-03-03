import numpy as np
import pandas as pd

def _stratified_indices(y, seed, n_train, n_val, n_test):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))

    pos = idx[y == 1]
    neg = idx[y == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    def take(arr, n1, n2, n3):
        return arr[:n1], arr[n1:n1+n2], arr[n1+n2:n1+n2+n3]

    # split each class then combine
    p_tr, p_va, p_te = take(pos,
                            int(len(pos)*n_train), int(len(pos)*n_val), int(len(pos)*n_test))
    n_tr, n_va, n_te = take(neg,
                            int(len(neg)*n_train), int(len(neg)*n_val), int(len(neg)*n_test))

    tr = np.concatenate([p_tr, n_tr])
    va = np.concatenate([p_va, n_va])
    te = np.concatenate([p_te, n_te])

    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

def baseline_split(df, label_col, seed=42, frac=(0.70, 0.15, 0.15), group_cols=None):
    """Random split. If group_cols provided, keeps groups together (best-effort)."""
    train_f, val_f, test_f = frac
    assert abs(train_f + val_f + test_f - 1.0) < 1e-9

    if group_cols:
        # group id as tuple -> string
        g = df[group_cols].astype(str).agg("||".join, axis=1)
        group_label = df.groupby(g)[label_col].max()
        groups = group_label.index.to_numpy()
        y = group_label.to_numpy()

        rng = np.random.default_rng(seed)
        order = np.arange(len(groups))
        rng.shuffle(order)

        # rough stratification by sorting groups by label blocks
        groups_pos = groups[y == 1]
        groups_neg = groups[y == 0]
        rng.shuffle(groups_pos); rng.shuffle(groups_neg)

        def split_groups(arr):
            n = len(arr)
            n_tr = int(n * train_f); n_va = int(n * val_f)
            return arr[:n_tr], arr[n_tr:n_tr+n_va], arr[n_tr+n_va:]

        tr_g = np.concatenate(split_groups(groups_pos)[0:1] + split_groups(groups_neg)[0:1])
        va_g = np.concatenate(split_groups(groups_pos)[1:2] + split_groups(groups_neg)[1:2])
        te_g = np.concatenate(split_groups(groups_pos)[2:3] + split_groups(groups_neg)[2:3])

        idx_train = df.index[g.isin(tr_g)].to_numpy()
        idx_val   = df.index[g.isin(va_g)].to_numpy()
        idx_test  = df.index[g.isin(te_g)].to_numpy()
        return idx_train, idx_val, idx_test

    # row-level stratified split
    n = len(df)
    n_train = train_f
    n_val = val_f
    n_test = test_f
    return _stratified_indices(df[label_col].values, seed, n_train, n_val, n_test)

def time_aware_split(df, label_col, time_col, frac=(0.70, 0.15, 0.15), group_cols=None):
    """Train on earlier time, test on later time. If group_cols provided, group by first time."""
    train_f, val_f, test_f = frac
    assert abs(train_f + val_f + test_f - 1.0) < 1e-9
    assert time_col in df.columns

    if group_cols:
        g = df[group_cols].astype(str).agg("||".join, axis=1)
        first_t = df.groupby(g)[time_col].min().sort_values()
        groups_sorted = first_t.index.to_numpy()

        n = len(groups_sorted)
        n_tr = int(n * train_f)
        n_va = int(n * val_f)

        tr_g = groups_sorted[:n_tr]
        va_g = groups_sorted[n_tr:n_tr+n_va]
        te_g = groups_sorted[n_tr+n_va:]

        idx_train = df.index[g.isin(tr_g)].to_numpy()
        idx_val   = df.index[g.isin(va_g)].to_numpy()
        idx_test  = df.index[g.isin(te_g)].to_numpy()
        return idx_train, idx_val, idx_test

    # row-level time split
    df_sorted = df.sort_values(time_col)
    idx = df_sorted.index.to_numpy()
    n = len(idx)
    n_tr = int(n * train_f)
    n_va = int(n * val_f)
    idx_train = idx[:n_tr]
    idx_val = idx[n_tr:n_tr+n_va]
    idx_test = idx[n_tr+n_va:]
    return idx_train, idx_val, idx_test