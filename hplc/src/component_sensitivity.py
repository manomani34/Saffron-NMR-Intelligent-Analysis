from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse.linalg import spsolve

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "outputs"
    / "independent_raw_matrices.npz"
)

MAPPING_FILE = (
    ROOT
    / "data"
    / "mapping"
    / "sample_mapping.csv"
)

OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "component_sensitivity_results.csv"
)


# ============================================================
# CONFIG
# ============================================================

COMPONENT_VALUES = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]

N_SPLITS = 2
N_REPEATS = 20

RANDOM_STATE = 42

ASLS_LAMBDA = 1e5
ASLS_P = 0.01


# ============================================================
# MISSING VALUES
# ============================================================

def fill_missing_linear(
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
    lam: float = ASLS_LAMBDA,
    p: float = ASLS_P,
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

        Z = W + lam * (
            D.T @ D
        )

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
# PREPROCESS ONE MATRIX
# ============================================================

def preprocess_matrix(
    X: np.ndarray,
) -> np.ndarray:

    processed = np.empty_like(
        X,
        dtype=float,
    )

    for i in range(
        X.shape[0]
    ):

        signal = fill_missing_linear(
            X[i]
        )

        baseline = asls_baseline(
            signal
        )

        corrected = (
            signal - baseline
        )

        processed[i] = snv(
            corrected
        )

    return processed


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    data = np.load(
        DATA_FILE
    )

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    if "SampleId" not in mapping.columns:
        raise ValueError(
            "SampleId column not found."
        )

    group_column = None

    for candidate in [
        "Group",
        "group",
        "Region",
        "region",
    ]:

        if candidate in mapping.columns:
            group_column = candidate
            break

    if group_column is None:
        raise ValueError(
            "Group column not found."
        )

    sample_ids = (
        data["SampleId"]
        .astype(int)
    )

    mapping = mapping.copy()

    mapping["SampleId"] = (
        mapping["SampleId"]
        .astype(int)
    )

    mapping = (
        mapping
        .set_index("SampleId")
        .loc[sample_ids]
        .reset_index()
    )

    y_text = (
        mapping[group_column]
        .astype(str)
        .to_numpy()
    )

    classes = np.unique(
        y_text
    )

    class_to_int = {
        name: i
        for i, name in enumerate(
            classes
        )
    }

    y = np.asarray(
        [
            class_to_int[name]
            for name in y_text
        ],
        dtype=int,
    )

    # --------------------------------------------------------
    # Preprocess each wavelength
    # --------------------------------------------------------

    print(
        "[1/3] Preprocessing 440 nm..."
    )

    X440 = preprocess_matrix(
        data["X_440"]
    )

    print(
        "[1/3] Preprocessing 250 nm..."
    )

    X250 = preprocess_matrix(
        data["X_250"]
    )

    print(
        "[1/3] Preprocessing 308 nm..."
    )

    X308 = preprocess_matrix(
        data["X_308"]
    )

    # Combined modality
    X = np.hstack(
        [
            X440,
            X250,
            X308,
        ]
    )

    return (
        X,
        y,
        classes,
        sample_ids,
    )


# ============================================================
# ONE-HOT
# ============================================================

def one_hot(
    y: np.ndarray,
    n_classes: int,
) -> np.ndarray:

    result = np.zeros(
        (
            len(y),
            n_classes,
        ),
        dtype=float,
    )

    result[
        np.arange(len(y)),
        y,
    ] = 1.0

    return result


# ============================================================
# PREDICTION
# ============================================================

def predict_classes(
    model: PLSRegression,
    X: np.ndarray,
) -> np.ndarray:

    scores = np.asarray(
        model.predict(X),
        dtype=float,
    )

    if scores.ndim == 1:
        scores = scores.reshape(
            -1,
            1,
        )

    return np.argmax(
        scores,
        axis=1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "HPLC PLS COMPONENT SENSITIVITY"
    )
    print("=" * 72)

    X, y, classes, sample_ids = (
        load_data()
    )

    print(
        f"Independent samples : {len(y)}"
    )

    print(
        f"Classes              : {len(classes)}"
    )

    print(
        f"Combined features    : {X.shape[1]}"
    )

    print(
        "Preprocessing         : AsLS + SNV"
    )

    print(
        "Modality              : Combined"
    )

    print(
        "CV                    : "
        f"Repeated Stratified "
        f"{N_SPLITS}-Fold × {N_REPEATS}"
    )

    print(
        "Components            : "
        + ", ".join(
            map(
                str,
                COMPONENT_VALUES,
            )
        )
    )

    print()

    # --------------------------------------------------------
    # Fixed CV splits
    # --------------------------------------------------------

    print(
        "[2/3] Creating fixed CV splits..."
    )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(
            X,
            y,
        )
    )

    print(
        f"Fixed splits          : {len(splits)}"
    )

    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "[3/3] Evaluating components..."
    )

    rows = []

    n_classes = len(
        classes
    )

    for n_components in (
        COMPONENT_VALUES
    ):

        print(
            f"Components = {n_components}"
        )

        for split_id, (
            train_idx,
            test_idx,
        ) in enumerate(
            splits,
            start=1,
        ):

            X_train_raw = X[
                train_idx
            ]

            X_test_raw = X[
                test_idx
            ]

            y_train = y[
                train_idx
            ]

            y_test = y[
                test_idx
            ]

            # ------------------------------------------------
            # Scaling FIT ONLY ON TRAIN
            # ------------------------------------------------

            scaler = StandardScaler()

            X_train = (
                scaler.fit_transform(
                    X_train_raw
                )
            )

            X_test = (
                scaler.transform(
                    X_test_raw
                )
            )

            # ------------------------------------------------
            # PLS-DA
            # ------------------------------------------------

            y_train_onehot = one_hot(
                y_train,
                n_classes,
            )

            max_components = min(
                n_components,
                X_train.shape[1],
                X_train.shape[0] - 1,
            )

            model = PLSRegression(
                n_components=max_components,
                scale=False,
                max_iter=2000,
                tol=1e-6,
            )

            model.fit(
                X_train,
                y_train_onehot,
            )

            y_pred = predict_classes(
                model,
                X_test,
            )

            rows.append(
                {
                    "Components":
                        n_components,

                    "Split":
                        split_id,

                    "Accuracy":
                        accuracy_score(
                            y_test,
                            y_pred,
                        ),

                    "BalancedAccuracy":
                        balanced_accuracy_score(
                            y_test,
                            y_pred,
                        ),

                    "MacroF1":
                        f1_score(
                            y_test,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        ),
                }
            )

        print(
            f"    completed "
            f"{len(splits)} splits"
        )

    detail = pd.DataFrame(
        rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        detail
        .groupby("Components")
        .agg(
            AccuracyMean=(
                "Accuracy",
                "mean",
            ),
            AccuracyStd=(
                "Accuracy",
                "std",
            ),
            BalancedAccuracyMean=(
                "BalancedAccuracy",
                "mean",
            ),
            BalancedAccuracyStd=(
                "BalancedAccuracy",
                "std",
            ),
            MacroF1Mean=(
                "MacroF1",
                "mean",
            ),
            MacroF1Std=(
                "MacroF1",
                "std",
            ),
            Splits=(
                "Split",
                "count",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print("=" * 72)
    print(
        "COMPONENT SENSITIVITY SUMMARY"
    )
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()