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
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
)
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

SUMMARY_FILE = (
    ROOT
    / "outputs"
    / "nested_component_summary.csv"
)

OUTER_DETAIL_FILE = (
    ROOT
    / "outputs"
    / "nested_component_outer_results.csv"
)

SELECTION_FILE = (
    ROOT
    / "outputs"
    / "nested_component_selection.csv"
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

OUTER_SPLITS = 2
OUTER_REPEATS = 20

INNER_SPLITS = 2

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
# PREPROCESSING
# ============================================================

def preprocess_matrix(
    X: np.ndarray,
) -> np.ndarray:

    result = np.empty_like(
        X,
        dtype=float,
    )

    for i in range(
        X.shape[0]
    ):

        y = fill_missing_linear(
            X[i]
        )

        baseline = asls_baseline(
            y
        )

        corrected = (
            y - baseline
        )

        result[i] = snv(
            corrected
        )

    return result


# ============================================================
# DATA
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
        for i, name in enumerate(classes)
    }

    y = np.asarray(
        [
            class_to_int[name]
            for name in y_text
        ],
        dtype=int,
    )

    print(
        "[1/4] Preprocessing 440 nm..."
    )

    X440 = preprocess_matrix(
        data["X_440"]
    )

    print(
        "[1/4] Preprocessing 250 nm..."
    )

    X250 = preprocess_matrix(
        data["X_250"]
    )

    print(
        "[1/4] Preprocessing 308 nm..."
    )

    X308 = preprocess_matrix(
        data["X_308"]
    )

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
# ONE HOT
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
# PREDICT
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
        "NESTED CV FOR PLS COMPONENT SELECTION"
    )
    print("=" * 72)

    (
        X,
        y,
        classes,
        sample_ids,
    ) = load_data()

    n_classes = len(
        classes
    )

    print(
        f"Independent samples : {len(y)}"
    )

    print(
        f"Classes              : {n_classes}"
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
        "Outer CV              : "
        f"Repeated Stratified "
        f"{OUTER_SPLITS}-Fold × "
        f"{OUTER_REPEATS}"
    )

    print(
        "Inner CV              : "
        f"KFold × {INNER_SPLITS}"
    )

    print(
        "Components             : "
        + ", ".join(
            map(
                str,
                COMPONENT_VALUES,
            )
        )
    )

    print()

    # ========================================================
    # OUTER SPLITS
    # ========================================================

    print(
        "[2/4] Creating outer CV splits..."
    )

    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )

    outer_splits = list(
        outer_cv.split(
            X,
            y,
        )
    )

    print(
        f"Outer splits          : "
        f"{len(outer_splits)}"
    )

    print()

    # ========================================================
    # NESTED CV
    # ========================================================

    print(
        "[3/4] Running nested CV..."
    )

    outer_rows = []
    selection_rows = []

    for outer_id, (
        outer_train_idx,
        outer_test_idx,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        X_outer_train = X[
            outer_train_idx
        ]

        X_outer_test = X[
            outer_test_idx
        ]

        y_outer_train = y[
            outer_train_idx
        ]

        y_outer_test = y[
            outer_test_idx
        ]

        # ----------------------------------------------------
        # Inner CV
        # ----------------------------------------------------

        inner_seed = (
            RANDOM_STATE
            + outer_id
        )

        inner_cv = KFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=inner_seed,
        )

        inner_scores = {
            component: []
            for component
            in COMPONENT_VALUES
        }

        # ----------------------------------------------------
        # Test each component inside INNER CV
        # ----------------------------------------------------

        for (
            inner_train_idx,
            inner_valid_idx,
        ) in inner_cv.split(
            X_outer_train
        ):

            X_inner_train_raw = (
                X_outer_train[
                    inner_train_idx
                ]
            )

            X_inner_valid_raw = (
                X_outer_train[
                    inner_valid_idx
                ]
            )

            y_inner_train = (
                y_outer_train[
                    inner_train_idx
                ]
            )

            y_inner_valid = (
                y_outer_train[
                    inner_valid_idx
                ]
            )

            # ------------------------------------------------
            # Scaling FIT ONLY on inner train
            # ------------------------------------------------

            scaler = StandardScaler()

            X_inner_train = (
                scaler.fit_transform(
                    X_inner_train_raw
                )
            )

            X_inner_valid = (
                scaler.transform(
                    X_inner_valid_raw
                )
            )

            y_inner_onehot = one_hot(
                y_inner_train,
                n_classes,
            )

            for component in (
                COMPONENT_VALUES
            ):

                actual_components = min(
                    component,
                    X_inner_train.shape[1],
                    X_inner_train.shape[0] - 1,
                )

                model = PLSRegression(
                    n_components=actual_components,
                    scale=False,
                    max_iter=2000,
                    tol=1e-6,
                )

                model.fit(
                    X_inner_train,
                    y_inner_onehot,
                )

                y_pred = (
                    predict_classes(
                        model,
                        X_inner_valid,
                    )
                )

                score = (
                    balanced_accuracy_score(
                        y_inner_valid,
                        y_pred,
                    )
                )

                inner_scores[
                    component
                ].append(
                    score
                )

        # ----------------------------------------------------
        # Select component using ONLY inner CV
        # ----------------------------------------------------

        inner_means = {
            component: float(
                np.mean(scores)
            )
            for component, scores
            in inner_scores.items()
        }

        selected_component = max(
            COMPONENT_VALUES,
            key=lambda c:
                inner_means[c],
        )

        selection_rows.append(
            {
                "OuterFold":
                    outer_id,

                "SelectedComponents":
                    selected_component,

                "SelectedInnerBA":
                    inner_means[
                        selected_component
                    ],
            }
        )

        # ----------------------------------------------------
        # Fit selected component on ENTIRE outer train
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_outer_train_scaled = (
            scaler.fit_transform(
                X_outer_train
            )
        )

        X_outer_test_scaled = (
            scaler.transform(
                X_outer_test
            )
        )

        y_outer_onehot = one_hot(
            y_outer_train,
            n_classes,
        )

        actual_components = min(
            selected_component,
            X_outer_train_scaled.shape[1],
            X_outer_train_scaled.shape[0] - 1,
        )

        model = PLSRegression(
            n_components=actual_components,
            scale=False,
            max_iter=2000,
            tol=1e-6,
        )

        model.fit(
            X_outer_train_scaled,
            y_outer_onehot,
        )

        y_pred = predict_classes(
            model,
            X_outer_test_scaled,
        )

        outer_rows.append(
            {
                "OuterFold":
                    outer_id,

                "SelectedComponents":
                    selected_component,

                "InnerBalancedAccuracy":
                    inner_means[
                        selected_component
                    ],

                "Accuracy":
                    accuracy_score(
                        y_outer_test,
                        y_pred,
                    ),

                "BalancedAccuracy":
                    balanced_accuracy_score(
                        y_outer_test,
                        y_pred,
                    ),

                "MacroF1":
                    f1_score(
                        y_outer_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),

                "OuterTrainSize":
                    len(outer_train_idx),

                "OuterTestSize":
                    len(outer_test_idx),
            }
        )

        print(
            f"    Outer fold "
            f"{outer_id}/{len(outer_splits)} "
            f"→ selected components = "
            f"{selected_component}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "[4/4] Creating nested CV summary..."
    )

    outer_df = pd.DataFrame(
        outer_rows
    )

    selection_df = pd.DataFrame(
        selection_rows
    )

    summary = pd.DataFrame(
        [
            {
                "AccuracyMean":
                    outer_df["Accuracy"].mean(),

                "AccuracyStd":
                    outer_df["Accuracy"].std(),

                "BalancedAccuracyMean":
                    outer_df[
                        "BalancedAccuracy"
                    ].mean(),

                "BalancedAccuracyStd":
                    outer_df[
                        "BalancedAccuracy"
                    ].std(),

                "MacroF1Mean":
                    outer_df[
                        "MacroF1"
                    ].mean(),

                "MacroF1Std":
                    outer_df[
                        "MacroF1"
                    ].std(),

                "OuterSplits":
                    len(outer_df),

                "MeanSelectedComponents":
                    selection_df[
                        "SelectedComponents"
                    ].mean(),

                "MedianSelectedComponents":
                    selection_df[
                        "SelectedComponents"
                    ].median(),
            }
        ]
    )

    # Selection frequency
    selection_frequency = (
        selection_df[
            "SelectedComponents"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "Components"
        )
        .reset_index(
            name="SelectionCount"
        )
    )

    selection_frequency[
        "SelectionRate"
    ] = (
        selection_frequency[
            "SelectionCount"
        ]
        / len(selection_df)
    )

    # ========================================================
    # SAVE
    # ========================================================

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    outer_df.to_csv(
        OUTER_DETAIL_FILE,
        index=False,
    )

    selection_df.to_csv(
        SELECTION_FILE,
        index=False,
    )

    selection_frequency.to_csv(
        ROOT
        / "outputs"
        / "nested_component_selection_frequency.csv",
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print("=" * 72)
    print(
        "NESTED CV SUMMARY"
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
        "Component selection frequency:"
    )

    print(
        selection_frequency.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print(
        f"Outer   : {OUTER_DETAIL_FILE}"
    )

    print(
        f"Selection : {SELECTION_FILE}"
    )


if __name__ == "__main__":
    main()