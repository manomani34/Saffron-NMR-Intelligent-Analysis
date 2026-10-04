from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve


@dataclass
class FeatureConfig:
    asls_lambda: float = 1e5
    asls_p: float = 0.01

    min_peak_prominence_ratio: float = 0.02
    min_peak_distance_points: int = 15

    top_n_peaks: int = 3


# ============================================================
# MISSING VALUES
# ============================================================

def _fill_missing_linear(
    y: np.ndarray,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    ).copy()

    finite = np.isfinite(y)

    if finite.all():
        return y

    if finite.sum() == 0:
        return np.zeros_like(y)

    indices = np.arange(
        len(y),
        dtype=float,
    )

    y[~finite] = np.interp(
        indices[~finite],
        indices[finite],
        y[finite],
    )

    return y


# ============================================================
# ASLS
# ============================================================

def asls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 20,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(y)

    if n < 3:
        return np.zeros_like(y)

    D = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )

    w = np.ones(n)

    for _ in range(n_iter):

        W = sparse.diags(
            w,
            0,
            shape=(n, n),
            format="csc",
        )

        Z = W + lam * (D.T @ D)

        baseline = spsolve(
            Z,
            w * y,
        )

        w = np.where(
            y > baseline,
            p,
            1.0 - p,
        )

    return baseline


# ============================================================
# SNV
# ============================================================

def snv(
    y: np.ndarray,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    mean = np.mean(y)
    std = np.std(y)

    if (
        not np.isfinite(mean)
        or not np.isfinite(std)
        or std <= 1e-12
    ):
        return np.zeros_like(y)

    return (
        y - mean
    ) / std


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_signal(
    y: np.ndarray,
    method: str,
    config: FeatureConfig,
) -> np.ndarray:

    y = _fill_missing_linear(y)

    if method == "Raw":
        return y.copy()

    if method == "SNV":
        return snv(y)

    if method == "AsLS":

        baseline = asls_baseline(
            y,
            lam=config.asls_lambda,
            p=config.asls_p,
        )

        return y - baseline

    if method == "AsLS + SNV":

        baseline = asls_baseline(
            y,
            lam=config.asls_lambda,
            p=config.asls_p,
        )

        corrected = y - baseline

        return snv(corrected)

    raise ValueError(
        f"Unknown preprocessing method: {method}"
    )


# ============================================================
# AREA
# ============================================================

def _safe_area(
    x: np.ndarray,
    t: np.ndarray,
) -> float:

    mask = (
        np.isfinite(x)
        & np.isfinite(t)
    )

    if mask.sum() < 2:
        return 0.0

    return float(
        trapezoid(
            x[mask],
            t[mask],
        )
    )


def _range_area(
    x: np.ndarray,
    t: np.ndarray,
    start: float,
    end: float,
) -> float:

    mask = (
        np.isfinite(x)
        & np.isfinite(t)
        & (t >= start)
        & (t <= end)
    )

    if mask.sum() < 2:
        return 0.0

    return float(
        trapezoid(
            x[mask],
            t[mask],
        )
    )


# ============================================================
# PEAK AREA
# ============================================================

def _peak_area(
    x: np.ndarray,
    t: np.ndarray,
    peak_index: int,
) -> float:

    n = len(x)

    left = peak_index
    right = peak_index

    while (
        left > 0
        and x[left - 1] <= x[left]
    ):
        left -= 1

    while (
        right < n - 1
        and x[right + 1] <= x[right]
    ):
        right += 1

    if right <= left:
        return 0.0

    return abs(
        _safe_area(
            x[left:right + 1],
            t[left:right + 1],
        )
    )


# ============================================================
# BASIC FEATURES
# ============================================================

def _basic_features(
    x: np.ndarray,
    t: np.ndarray,
    prefix: str,
) -> dict[str, float]:

    mask = (
        np.isfinite(x)
        & np.isfinite(t)
    )

    if mask.sum() < 2:
        return {}

    x = x[mask]
    t = t[mask]

    total_area = _safe_area(
        x,
        t,
    )

    absolute_area = _safe_area(
        np.abs(x),
        t,
    )

    positive_area = _safe_area(
        np.maximum(
            x,
            0.0,
        ),
        t,
    )

    max_idx = int(
        np.argmax(x)
    )

    # --------------------------------------------------------
    # Three biologically useful broad regions
    # --------------------------------------------------------

    early = abs(
        _range_area(
            x,
            t,
            0.0,
            10.0,
        )
    )

    middle = abs(
        _range_area(
            x,
            t,
            10.0,
            14.0,
        )
    )

    late = abs(
        _range_area(
            x,
            t,
            14.0,
            30.0,
        )
    )

    total_abs = (
        early
        + middle
        + late
        + 1e-12
    )

    # --------------------------------------------------------
    # Center of area
    # --------------------------------------------------------

    abs_x = np.abs(x)

    abs_area = _safe_area(
        abs_x,
        t,
    )

    if abs_area > 1e-12:

        retention_center = float(
            trapezoid(
                t * abs_x,
                t,
            )
            / abs_area
        )

    else:
        retention_center = 0.0

    features = {

        # Global shape
        f"{prefix}_area":
            total_area,

        f"{prefix}_abs_area":
            absolute_area,

        f"{prefix}_positive_area":
            positive_area,

        f"{prefix}_max":
            float(np.max(x)),

        f"{prefix}_std":
            float(np.std(x)),

        f"{prefix}_range":
            float(
                np.max(x)
                - np.min(x)
            ),

        f"{prefix}_tmax":
            float(t[max_idx]),

        f"{prefix}_retention_center":
            retention_center,

        # Broad retention regions
        f"{prefix}_area_early":
            early,

        f"{prefix}_area_middle":
            middle,

        f"{prefix}_area_late":
            late,

        # Region fractions
        f"{prefix}_fraction_early":
            early / total_abs,

        f"{prefix}_fraction_middle":
            middle / total_abs,

        f"{prefix}_fraction_late":
            late / total_abs,
    }

    return features


# ============================================================
# PEAK FEATURES
# ============================================================

def _peak_features(
    x: np.ndarray,
    t: np.ndarray,
    prefix: str,
    config: FeatureConfig,
) -> dict[str, float]:

    mask = (
        np.isfinite(x)
        & np.isfinite(t)
    )

    x = x[mask]
    t = t[mask]

    if len(x) < 3:

        return {
            f"{prefix}_peak_count": 0.0
        }

    amplitude = float(
        np.max(x)
        - np.min(x)
    )

    prominence_threshold = max(
        amplitude
        * config.min_peak_prominence_ratio,
        1e-8,
    )

    peaks, properties = find_peaks(
        x,
        prominence=prominence_threshold,
        distance=config.min_peak_distance_points,
    )

    result = {
        f"{prefix}_peak_count":
            float(len(peaks))
    }

    if len(peaks) == 0:

        for i in range(
            config.top_n_peaks
        ):

            result[
                f"{prefix}_peak_{i+1}_height"
            ] = 0.0

            result[
                f"{prefix}_peak_{i+1}_time"
            ] = 0.0

            result[
                f"{prefix}_peak_{i+1}_area"
            ] = 0.0

        return result

    heights = x[peaks]

    order = np.argsort(
        heights
    )[::-1]

    peaks = peaks[order]
    heights = heights[order]

    for i in range(
        config.top_n_peaks
    ):

        if i < len(peaks):

            peak = int(
                peaks[i]
            )

            result[
                f"{prefix}_peak_{i+1}_height"
            ] = float(
                heights[i]
            )

            result[
                f"{prefix}_peak_{i+1}_time"
            ] = float(
                t[peak]
            )

            result[
                f"{prefix}_peak_{i+1}_area"
            ] = _peak_area(
                x,
                t,
                peak,
            )

        else:

            result[
                f"{prefix}_peak_{i+1}_height"
            ] = 0.0

            result[
                f"{prefix}_peak_{i+1}_time"
            ] = 0.0

            result[
                f"{prefix}_peak_{i+1}_area"
            ] = 0.0

    return result


# ============================================================
# SINGLE CHROMATOGRAM
# ============================================================

def extract_chromatogram_features(
    x: np.ndarray,
    time: np.ndarray,
    prefix: str,
    config: FeatureConfig | None = None,
) -> dict[str, float]:

    config = (
        config
        or FeatureConfig()
    )

    # --------------------------------------------------------
    # Important:
    # V2 currently uses AsLS + SNV because this was the
    # stronger preprocessing family in the previous test.
    # --------------------------------------------------------

    x = preprocess_signal(
        x,
        "AsLS + SNV",
        config,
    )

    features = {}

    features.update(
        _basic_features(
            x,
            time,
            prefix,
        )
    )

    features.update(
        _peak_features(
            x,
            time,
            prefix,
            config,
        )
    )

    return features


# ============================================================
# MATRIX
# ============================================================

def build_feature_matrix(
    X: np.ndarray,
    time: np.ndarray,
    prefix: str,
    config: FeatureConfig | None = None,
) -> tuple[
    np.ndarray,
    list[str],
]:

    config = (
        config
        or FeatureConfig()
    )

    rows = [
        extract_chromatogram_features(
            X[i],
            time,
            prefix,
            config,
        )
        for i in range(
            X.shape[0]
        )
    ]

    names = list(
        rows[0].keys()
    )

    matrix = np.asarray(
        [
            [
                row[name]
                for name in names
            ]
            for row in rows
        ],
        dtype=float,
    )

    return (
        matrix,
        names,
    )


# ============================================================
# COMBINED
# ============================================================

def build_combined_feature_matrix(
    X_440: np.ndarray,
    X_250: np.ndarray,
    X_308: np.ndarray,
    time_440: np.ndarray,
    time_250: np.ndarray,
    time_308: np.ndarray,
    config: FeatureConfig | None = None,
) -> tuple[
    np.ndarray,
    list[str],
]:

    config = (
        config
        or FeatureConfig()
    )

    f440, n440 = build_feature_matrix(
        X_440,
        time_440,
        "440",
        config,
    )

    f250, n250 = build_feature_matrix(
        X_250,
        time_250,
        "250",
        config,
    )

    f308, n308 = build_feature_matrix(
        X_308,
        time_308,
        "308",
        config,
    )

    combined = np.hstack(
        [
            f440,
            f250,
            f308,
        ]
    )

    return (
        combined,
        n440 + n250 + n308,
    )


if __name__ == "__main__":

    print(
        "Feature Engineering V2 loaded successfully."
    )