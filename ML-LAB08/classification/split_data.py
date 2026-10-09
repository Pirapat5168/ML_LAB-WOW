
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

NAMES = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]


def split_data(X, y, val_size=0.15, test_size=0.15, seed=42):
    """70/15/15 by default."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size),
        stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(out_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    out_dir = Path(out_dir)
    for name, arr in zip(NAMES, [X_train, X_val, X_test, y_train, y_val, y_test]):
        np.save(out_dir / f"{name}.npy", arr)


def load_splits(out_dir):
    out_dir = Path(out_dir)
    return tuple(np.load(out_dir / f"{name}.npy") for name in NAMES)
