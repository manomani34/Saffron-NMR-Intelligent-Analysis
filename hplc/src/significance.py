from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .feature_engineering import (
    asls_baseline,
    snv,
)


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
    / "permutation_test_results.csv"
)


# ============================================================
# CONFIG
# ============================================================

PLS_COMPONENTS = 10

N_SPLITS = 2

DEFAULT_REPEATS = 20
DEFAULT_PERMUTATIONS = 500

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

    idx = np.arange(
        len(y),
        dtype=float,
    )

    y[~finite] = np.interp(
        idx[~finite],
        idx[finite],
        y[finite],
    )

    return y


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
            y,
            lam=ASLS_LAMBDA,
            p=ASLS_P,
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

def find_group_column(
    mapping: pd.DataFrame,
) -> str:

    for candidate in (
        "Group",
        "group",
        "Region",
        "region",
    ):

        if candidate in mapping.columns:
            return candidate

    raise ValueError(
        "Group column not found in mapping file."
    )


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

    group_column = find_group_column(
        mapping
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
# FIXED 11-CLASS BALANCED ACCURACY
# ============================================================

def fixed_class_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> float:
    """
    Compute balanced accuracy explicitly across all classes.

    All classes 0..n_classes-1 are included.
    Because predictions from all CV folds are pooled first,
    the metric does not depend on whether a particular class
    happens to be absent from an individual validation fold.
    """

    recalls = []

    for class_id in range(
        n_classes
    ):

        mask = (
            y_true == class_id
        )

        class_count = int(
            mask.sum()
        )

        if class_count == 0:
            recall = 0.0
        else:
            correct = np.sum(
                y_pred[mask]
                == class_id
            )

            recall = (
                float(correct)
                / float(class_count)
            )

        recalls.append(
            recall
        )

    return float(
        np.mean(recalls)
    )


# ============================================================
# CV SCORE
# ============================================================

def evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits,
    n_classes: int,
) -> tuple[float, int]:

    all_true = []
    all_pred = []

    convergence_warnings = 0

    for train_idx, test_idx in splits:

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

        # ----------------------------------------------------
        # Scaling
        # FIT ONLY ON TRAIN
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # One-hot target
        # ----------------------------------------------------

        y_train_onehot = one_hot(
            y_train,
            n_classes,
        )

        n_components = min(
            PLS_COMPONENTS,
            X_train.shape[1],
            X_train.shape[0] - 1,
        )

        model = PLSRegression(
            n_components=n_components,
            scale=False,
            max_iter=2000,
            tol=1e-6,
        )

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        with warnings.catch_warnings(
            record=True
        ) as caught:

            warnings.simplefilter(
                "always"
            )

            model.fit(
                X_train,
                y_train_onehot,
            )

        convergence_warnings += sum(
            1
            for warning in caught
            if issubclass(
                warning.category,
                ConvergenceWarning,
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = predict_classes(
            model,
            X_test,
        )

        # ----------------------------------------------------
        # Pool ALL OOF predictions
        # ----------------------------------------------------

        all_true.extend(
            y_test.tolist()
        )

        all_pred.extend(
            y_pred.tolist()
        )

    # ========================================================
    # POOLED FIXED-CLASS BA
    # ========================================================

    all_true = np.asarray(
        all_true,
        dtype=int,
    )

    all_pred = np.asarray(
        all_pred,
        dtype=int,
    )

    balanced_accuracy = (
        fixed_class_balanced_accuracy(
            all_true,
            all_pred,
            n_classes,
        )
    )

    return (
        balanced_accuracy,
        convergence_warnings,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
    )

    args = parser.parse_args()

    permutations = int(
        args.permutations
    )

    repeats = int(
        args.repeats
    )

    if permutations < 1:
        raise ValueError(
            "permutations must be >= 1"
        )

    if repeats < 1:
        raise ValueError(
            "repeats must be >= 1"
        )

    print("=" * 72)
    print(
        "HPLC SINGLE-SCENARIO PERMUTATION TEST"
    )
    print("=" * 72)

    X, y, classes, sample_ids = (
        load_data()
    )

    n_samples = len(y)
    n_classes = len(classes)

    print(
        f"Independent samples : {n_samples}"
    )

    print(
        f"Classes              : {n_classes}"
    )

    print(
        "Scenario             : "
        "AsLS + SNV | Combined"
    )

    print(
        f"PLS components        : "
        f"{PLS_COMPONENTS}"
    )

    print(
        "CV                   : "
        f"Repeated Stratified "
        f"{N_SPLITS}-Fold × {repeats}"
    )

    print(
        f"Permutations          : "
        f"{permutations}"
    )

    print()

    # ========================================================
    # FIXED CV SPLITS
    # ========================================================

    print(
        "[2/4] Creating fixed CV splits..."
    )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(
            np.zeros(
                n_samples
            ),
            y,
        )
    )

    print(
        f"Fixed splits          : "
        f"{len(splits)}"
    )

    print()

    # ========================================================
    # OBSERVED SCORE
    # ========================================================

    print(
        "[3/4] Computing observed score..."
    )

    observed_ba, observed_warnings = (
        evaluate_cv(
            X,
            y,
            splits,
            n_classes,
        )
    )

    print(
        f"Observed BA          : "
        f"{observed_ba:.6f}"
    )

    print(
        f"Convergence warnings : "
        f"{observed_warnings}"
    )

    print()

    # ========================================================
    # PERMUTATION NULL
    # ========================================================

    print(
        "[4/4] Computing permutation null..."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    null_scores = []

    total_warning_count = (
        observed_warnings
    )

    for permutation_id in range(
        1,
        permutations + 1,
    ):

        y_perm = rng.permutation(
            y
        )

        score, warning_count = (
            evaluate_cv(
                X,
                y_perm,
                splits,
                n_classes,
            )
        )

        null_scores.append(
            score
        )

        total_warning_count += (
            warning_count
        )

        if (
            permutation_id == 1
            or permutation_id % 10 == 0
            or permutation_id == permutations
        ):

            print(
                f"    Permutation "
                f"{permutation_id}/"
                f"{permutations}"
            )

    null = np.asarray(
        null_scores,
        dtype=float,
    )

    # ========================================================
    # P-VALUE
    # ========================================================

    exceed_count = int(
        np.sum(
            null >= observed_ba
        )
    )

    permutation_p = (
        exceed_count + 1
    ) / (
        permutations + 1
    )

    # Only one scenario is tested.
    # Therefore familywise p-value is identical.
    familywise_p = permutation_p

    null_mean = float(
        np.mean(null)
    )

    null_std = float(
        np.std(
            null,
            ddof=1,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = pd.DataFrame(
        [
            {
                "Scenario":
                    "AsLS + SNV | Combined",

                "PLSComponents":
                    PLS_COMPONENTS,

                "ObservedBalancedAccuracy":
                    observed_ba,

                "NullMean":
                    null_mean,

                "NullStd":
                    null_std,

                "PermutationPValue":
                    permutation_p,

                "FamilywisePValue":
                    familywise_p,

                "Permutations":
                    permutations,

                "CVRepeats":
                    repeats,

                "FixedSplits":
                    len(splits),

                "TotalConvergenceWarnings":
                    total_warning_count,
            }
        ]
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print("=" * 72)
    print(
        "PERMUTATION TEST RESULT"
    )
    print("=" * 72)

    print(
        results.to_string(
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