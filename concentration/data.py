"""Experiment 4 dataset definitions.

Kept local to Experiment 4 so the experiment is reproducible and
does not depend on unrelated preprocessing in the repository loader.
"""

import numpy as np
from sklearn.datasets import make_circles, make_moons


DATASETS = ("moons", "circles", "xor", "checkerboard")


def _make_checkerboard(n_samples=400, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n_samples, 2))
    y = (
        (np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2
    ).astype(int)
    return X, y


def _make_moons(n_samples=400, seed=42):
    return make_moons(
        n_samples=n_samples,
        noise=0.1,
        random_state=seed,
    )


def _make_circles(n_samples=400, seed=42):
    return make_circles(
        n_samples=n_samples,
        noise=0.15,
        factor=0.4,
        random_state=seed,
    )


def _make_xor(n_samples=400, seed=42):
    rng = np.random.RandomState(seed)
    n = n_samples // 4

    X = np.vstack([
        rng.randn(n, 2) * 0.3 + [1, 1],
        rng.randn(n, 2) * 0.3 + [-1, -1],
        rng.randn(n, 2) * 0.3 + [1, -1],
        rng.randn(n, 2) * 0.3 + [-1, 1],
    ])

    y = np.array([0] * n + [0] * n + [1] * n + [1] * n)
    return X, y


_LOADERS = {
    "moons": _make_moons,
    "circles": _make_circles,
    "xor": _make_xor,
    "checkerboard": _make_checkerboard,
}


def load_dataset(name, n_samples=400):
    try:
        return _LOADERS[name](n_samples=n_samples)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc
