import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, *extra_arrays, test_size=0.2):
    """Split X, y (and optionally other same-length arrays, e.g. raw
    images kept only for display) into train/test using one consistent
    split so every array stays aligned by index.

    Returns X_train, X_test, y_train, y_test, [extra1_train, extra1_test, ...]
    """

    # y must be an array, not a list, for stratify to work
    y = np.asarray(y)

    result = train_test_split(
        X, y, *extra_arrays,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    return result
