from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded


def interpolate_missing(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    n, m = X.shape
    base_x = np.arange(m, dtype=float)

    for i in range(n):
        row = X[i]
        mask = np.isfinite(row)
        if mask.all():
            continue
        if not mask.any():
            X[i] = 0.0
            continue
        if mask.sum() == 1:
            X[i, ~mask] = row[mask][0]
            continue
        X[i, ~mask] = np.interp(
            base_x[~mask],
            base_x[mask],
            row[mask],
        )

    return X


def snv(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def _asls_single(
    y: np.ndarray,
    lam: float,
    p: float,
    iterations: int,
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.

    The system matrix is symmetric pentadiagonal, so a banded solver is used
    instead of constructing a new sparse matrix and factorization at every
    iteration. The resulting solution is numerically equivalent to the
    standard D.T @ D formulation while being substantially faster for the
    repeated HPLC pipeline.
    """
    y = np.asarray(y, dtype=float)
    length = y.size

    if length < 3:
        return y.copy()

    # Second-difference penalty D.T @ D has the following diagonals:
    # main:  [1, 5, 6, ..., 6, 5, 1]
    # first: [-2, -4, ..., -4, -2]
    # second:[1, 1, ..., 1]
    main_penalty = np.full(length, 6.0, dtype=float)
    main_penalty[0] = 1.0
    main_penalty[1] = 5.0
    main_penalty[-2] = 5.0
    main_penalty[-1] = 1.0

    first_penalty = np.full(length - 1, -4.0, dtype=float)
    first_penalty[0] = -2.0
    first_penalty[-1] = -2.0

    second_penalty = np.ones(length - 2, dtype=float)

    weights = np.ones(length, dtype=float)

    # Upper-band representation required by scipy.linalg.solveh_banded:
    # row 0 -> second superdiagonal
    # row 1 -> first superdiagonal
    # row 2 -> main diagonal
    bands = np.zeros((3, length), dtype=float)

    for _ in range(iterations):
        bands[2] = weights + lam * main_penalty
        bands[1, 1:] = lam * first_penalty
        bands[0, 2:] = lam * second_penalty

        baseline = solveh_banded(
            bands,
            weights * y,
            lower=False,
            check_finite=False,
        )

        weights = np.where(
            y > baseline,
            p,
            1.0 - p,
        )

    return y - baseline


def asls_baseline(
    X: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    iterations: int = 10,
) -> np.ndarray:
    X = interpolate_missing(X)
    return np.vstack(
        [
            _asls_single(
                row,
                lam=lam,
                p=p,
                iterations=iterations,
            )
            for row in X
        ]
    )


def apply_preprocessing(
    X: np.ndarray,
    method: str,
    asls_lambda: float = 1e6,
    asls_p: float = 0.01,
    asls_iterations: int = 10,
) -> np.ndarray:
    method = method.lower().strip()
    X = interpolate_missing(X)

    if method == "raw":
        return X

    if method == "snv":
        return snv(X)

    if method == "asls_snv":
        return snv(
            asls_baseline(
                X,
                lam=asls_lambda,
                p=asls_p,
                iterations=asls_iterations,
            )
        )

    if method == "asls_l2":
        return l2_normalize(
            asls_baseline(
                X,
                lam=asls_lambda,
                p=asls_p,
                iterations=asls_iterations,
            )
        )

    raise ValueError(f"Unknown preprocessing method: {method}")
