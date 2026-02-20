from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split


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


def get_or_create_split(
    *,
    y: np.ndarray,
    seed: int,
    split_path: Path,
    dataset_path: Optional[Path] = None,
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

        if meta_n != expected_n:
            raise ValueError(f"Split file was created for n={meta_n}, but dataset now has n={expected_n}.")
        if dataset_hash and meta_hash and meta_hash != dataset_hash:
            raise ValueError("Dataset hash changed since split was created. (CSV changed or replaced.)")

        return train_idx, val_idx, test_idx

    train_idx, val_idx, test_idx = make_stratified_70_15_15_indices(y, seed)

    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=seed,
        n=expected_n,
        split="70/15/15",
        created_at=datetime.now().isoformat(timespec="seconds"),
        dataset_sha256=dataset_hash,
    )
    return train_idx, val_idx, test_idx