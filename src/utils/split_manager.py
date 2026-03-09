from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Literal, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

SplitMethod = Literal["stratified", "time"]

def project_root() -> Path:
    # .../fraud-dissertation/src/utils/split_manager.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_stratified_70_15_15_indices(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    idx = np.arange(n)

    idx_temp, idx_test = train_test_split(
        idx, test_size=0.15, random_state=seed, stratify=y
    )

    val_share_of_temp = 0.15 / 0.85
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_share_of_temp, random_state=seed, stratify=y[idx_temp]
    )

    # sorting is optional, but makes files/diffs stable
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)

def make_time_aware_70_15_15_indices(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-aware split: earlier -> later.
    Avoids splitting identical timestamps across boundaries (best-effort).
    """
    t = np.asarray(t)
    if t.ndim != 1:
        raise ValueError("t must be 1D.")
    idx_sorted = np.argsort(t, kind="mergesort")
    n = len(idx_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    idx_train = idx_sorted[:n_train]
    idx_val = idx_sorted[n_train:n_train + n_val]
    idx_test = idx_sorted[n_train + n_val:]
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def pd_isna(x) -> np.ndarray:
    # tiny helper to avoid importing pandas in utils
    # works for floats/ints/objects reasonably
    try:
        return np.isnan(x)
    except Exception:
        return np.asarray([v is None for v in x])


# ---------- Leakage checks ----------

def assert_no_id_overlap(ids: Sequence, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """
    ids can be TransactionID or a composite like (card1, card2, addr1).
    If you pass a tuple/list per row, convert it before calling.
    """
    ids = np.asarray(ids).astype(str)

    tr = set(ids[train_idx].tolist())
    va = set(ids[val_idx].tolist())
    te = set(ids[test_idx].tolist())

    if not tr.isdisjoint(va):
        raise ValueError("Leakage: ID overlap between train and val.")
    if not tr.isdisjoint(te):
        raise ValueError("Leakage: ID overlap between train and test.")
    if not va.isdisjoint(te):
        raise ValueError("Leakage: ID overlap between val and test.")

def assert_time_order(t: Sequence, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    t = np.asarray(t)
    tr_max = t[train_idx].max()
    va_min = t[val_idx].min()
    va_max = t[val_idx].max()
    te_min = t[test_idx].min()

    if tr_max > va_min:
        raise ValueError(f"Time leakage: train max ({tr_max}) > val min ({va_min})")
    if va_max > te_min:
        raise ValueError(f"Time leakage: val max ({va_max}) > test min ({te_min})")


def get_or_create_split(
    *,
    y: np.ndarray,
    seed: int,
    split_path: Path,
    dataset_path: Optional[Path] = None,
    method: SplitMethod = "stratified",
    t: Optional[np.ndarray] = None,
    ids_for_overlap_check: Optional[Sequence] = None,
    run_checks: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path.parent.mkdir(parents=True, exist_ok=True)

    expected_n = len(y)
    dataset_hash = sha256_file(dataset_path) if dataset_path else ""

    if split_path.exists():
        data = np.load(split_path, allow_pickle=True)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]

        meta_n = int(data["n"])
        meta_hash = str(data["dataset_sha256"])
        meta_method = str(data["method"]) if "method" in data else "stratified"

        if meta_n != expected_n:
            raise ValueError(f"Split file was created for n={meta_n}, but dataset now has n={expected_n}.")
        if dataset_hash and meta_hash and meta_hash != dataset_hash:
            raise ValueError("Dataset hash changed since split was created. (CSV changed or replaced.)")
        if meta_method != method:
            raise ValueError(f"Split file method is '{meta_method}', but you requested '{method}'. Use a new split_path.")

        return train_idx, val_idx, test_idx

    if method == "stratified":
        train_idx, val_idx, test_idx = make_stratified_70_15_15_indices(y, seed)
    elif method == "time":
        if t is None:
            raise ValueError("method='time' requires t=... (time/order array)")
        train_idx, val_idx, test_idx = make_time_aware_70_15_15_indices(t)
    else:
        raise ValueError(f"Unknown method: {method}")

    if run_checks:
        if ids_for_overlap_check is not None:
            assert_no_id_overlap(ids_for_overlap_check, train_idx, val_idx, test_idx)
        if method == "time" and t is not None:
            assert_time_order(t, train_idx, val_idx, test_idx)

    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=seed,
        n=expected_n,
        split="70/15/15",
        method=method,
        created_at=datetime.now().isoformat(timespec="seconds"),
        dataset_sha256=dataset_hash,
    )
    return train_idx, val_idx, test_idx