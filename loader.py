import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _make_checkerboard(n_samples, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2, 2, (n_samples, 2))
    y = ((np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2).astype(int)
    return x, y


def _make_moons_dataset(n_samples, seed=42):
    return make_moons(
        n_samples=n_samples,
        noise=0.1,
        random_state=seed,
    )


def _make_circles_dataset(n_samples, seed=42):
    return make_circles(
        n_samples=n_samples,
        noise=0.15,
        factor=0.4,
        random_state=seed,
    )


def _make_xor(n_samples, seed=42):
    rng = np.random.RandomState(seed)

    n = n_samples // 4

    x = np.vstack(
        [
            rng.randn(n, 2) * 0.3 + [1, 1],
            rng.randn(n, 2) * 0.3 + [-1, -1],
            rng.randn(n, 2) * 0.3 + [1, -1],
            rng.randn(n, 2) * 0.3 + [-1, 1],
        ]
    )

    y = np.array(
        [0] * n
        + [0] * n
        + [1] * n
        + [1] * n
    )

    return x, y


def load_dataset(dataset_name, n_samples=400, n_qubits=4):
    """
    Load, preprocess and normalize a synthetic dataset.
    """

    seed = 42

    if dataset_name == "checkerboard":
        x, y = _make_checkerboard(n_samples, seed)

    elif dataset_name == "moons":
        x, y = _make_moons_dataset(n_samples, seed)

    elif dataset_name == "circles":
        x, y = _make_circles_dataset(n_samples, seed)

    elif dataset_name == "xor":
        x, y = _make_xor(n_samples, seed)

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if isinstance(y, pd.Series):
        y = y.to_numpy()

    if y.dtype == object or y.dtype.kind == "U":
        y = LabelEncoder().fit_transform(y)

    y = y.astype(int)

    categorical_columns = [
        i
        for i in range(x.shape[1])
        if x[:, i].dtype == object or x[:, i].dtype.kind == "U"
    ]

    if categorical_columns:
        numeric_columns = [
            i
            for i in range(x.shape[1])
            if i not in categorical_columns
        ]

        if numeric_columns:
            x_numeric = x[:, numeric_columns].astype(np.float64)
        else:
            x_numeric = np.empty((x.shape[0], 0))

        encoded_columns = []

        for column in categorical_columns:
            encoded = LabelEncoder().fit_transform(x[:, column])
            encoded_columns.append(encoded.reshape(-1, 1))

        x_categorical = (
            np.hstack(encoded_columns)
            if encoded_columns
            else np.empty((x.shape[0], 0))
        )

        x = (
            np.hstack([x_numeric, x_categorical])
            if x_categorical.shape[1] > 0
            else x_numeric
        )

    else:
        x = x.astype(np.float64)

    mask = ~(np.isnan(x).any(axis=1) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' is empty after removing NaNs."
        )

    x = StandardScaler().fit_transform(x)

    if x.shape[1] > n_qubits:
        pca = PCA(
            n_components=n_qubits,
            random_state=42,
        )

        x = pca.fit_transform(x)

        explained_variance = np.sum(
            pca.explained_variance_ratio_
        )

        if explained_variance < 0.80:
            warnings.warn(
                f"PCA retains only "
                f"{explained_variance * 100:.2f}% "
                "of the variance."
            )

    x = x - x.mean(axis=0)

    scale = np.abs(x).max(axis=0)
    scale[scale == 0] = 1

    x = (x / scale) * np.pi

    return x, y
