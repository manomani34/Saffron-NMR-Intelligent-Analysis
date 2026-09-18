from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
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

DATA_FILE = ROOT / "outputs" / "independent_raw_matrices.npz"

MAPPING_FILE = (
    ROOT / "data" / "mapping" / "sample_mapping.csv"
)

SUMMARY_FILE = (
    ROOT / "outputs" / "candidate10_summary.csv"
)

PREDICTIONS_FILE = (
    ROOT / "outputs" / "candidate10_predictions.csv"
)

GROUP_FILE = (
    ROOT / "outputs" / "candidate10_group_performance.csv"
)

SAMPLE_FILE = (
    ROOT / "outputs" / "candidate10_sample_stability.csv"
)


# ============================================================
# CONFIG
# ============================================================

PLS_COMPONENTS = 10

N_SPLITS = 2
N_REPEATS = 20

RANDOM_STATE = 42

ASLS_LAMBDA = 1e5
ASLS_P = 0.01


# ============================================================
# HELPERS
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

        corrected = y - baseline

        result[i] = snv(
            corrected
        )

    return result


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
# LOAD DATA
# ============================================================

def load_data():

    data = np.load(
        DATA_FILE
    )

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    sample_ids = (
        data["SampleId"]
        .astype(int)
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

    group_text = (
        mapping[group_column]
        .astype(str)
        .to_numpy()
    )

    classes = np.unique(
        group_text
    )

    class_to_int = {
        name: i
        for i, name in enumerate(classes)
    }

    y = np.asarray(
        [
            class_to_int[name]
            for name in group_text
        ],
        dtype=int,
    )

    # --------------------------------------------------------
    # Three wavelengths
    # --------------------------------------------------------

    X440 = preprocess_matrix(
        data["X_440"]
    )

    X250 = preprocess_matrix(
        data["X_250"]
    )

    X308 = preprocess_matrix(
        data["X_308"]
    )

    # Combined
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
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "HPLC CANDIDATE 10-COMPONENT EVALUATION"
    )
    print("=" * 72)

    X, y, classes, sample_ids = (
        load_data()
    )

    n_classes = len(classes)

    print(
        f"Independent samples : {len(y)}"
    )

    print(
        f"Classes              : {n_classes}"
    )

    print(
        f"Combined variables   : {X.shape[1]}"
    )

    print(
        "Preprocessing         : AsLS + SNV"
    )

    print(
        "Modality              : Combined"
    )

    print(
        f"PLS components        : {PLS_COMPONENTS}"
    )

    print(
        f"CV                   : Repeated Stratified "
        f"{N_SPLITS}-Fold × {N_REPEATS}"
    )

    print()

    # ========================================================
    # FIXED CV
    # ========================================================

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    splits = list(
        cv.split(
            np.zeros(len(y)),
            y,
        )
    )

    print(
        f"Fixed splits          : {len(splits)}"
    )

    print()

    # ========================================================
    # EVALUATION
    # ========================================================

    prediction_rows = []
    fold_rows = []

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

        # ----------------------------------------------------
        # Scaling: FIT ONLY ON TRAIN
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
        # PLS-DA
        # ----------------------------------------------------

        model = PLSRegression(
            n_components=min(
                PLS_COMPONENTS,
                X_train.shape[1],
                X_train.shape[0] - 1,
            ),
            scale=False,
            max_iter=2000,
            tol=1e-6,
        )

        model.fit(
            X_train,
            one_hot(
                y_train,
                n_classes,
            ),
        )

        scores = np.asarray(
            model.predict(X_test),
            dtype=float,
        )

        if scores.ndim == 1:
            scores = scores.reshape(
                -1,
                1,
            )

        y_pred = np.argmax(
            scores,
            axis=1,
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_rows.append(
            {
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

        # ----------------------------------------------------
        # Prediction details
        # ----------------------------------------------------

        for local_pos, original_index in enumerate(
            test_idx
        ):

            actual = int(
                y_test[local_pos]
            )

            predicted = int(
                y_pred[local_pos]
            )

            prediction_rows.append(
                {
                    "Split":
                        split_id,

                    "SampleId":
                        int(
                            sample_ids[
                                original_index
                            ]
                        ),

                    "ActualClass":
                        str(
                            classes[actual]
                        ),

                    "PredictedClass":
                        str(
                            classes[predicted]
                        ),

                    "Correct":
                        bool(
                            actual == predicted
                        ),

                    "PredictedScore":
                        float(
                            np.max(
                                scores[
                                    local_pos
                                ]
                            )
                        ),

                    "ScoreMargin":
                        float(
                            np.sort(
                                scores[
                                    local_pos
                                ]
                            )[-1]
                            -
                            np.sort(
                                scores[
                                    local_pos
                                ]
                            )[-2]
                        )
                        if scores.shape[1] >= 2
                        else 0.0,
                }
            )

        if (
            split_id == 1
            or split_id % 10 == 0
            or split_id == len(splits)
        ):
            print(
                f"    Split "
                f"{split_id}/{len(splits)}"
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    predictions = pd.DataFrame(
        prediction_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "Model":
                    "PLS-DA",

                "Preprocessing":
                    "AsLS + SNV",

                "Modality":
                    "Combined",

                "PLSComponents":
                    PLS_COMPONENTS,

                "Samples":
                    len(y),

                "Classes":
                    n_classes,

                "Variables":
                    X.shape[1],

                "AccuracyMean":
                    folds["Accuracy"].mean(),

                "AccuracyStd":
                    folds["Accuracy"].std(),

                "BalancedAccuracyMean":
                    folds[
                        "BalancedAccuracy"
                    ].mean(),

                "BalancedAccuracyStd":
                    folds[
                        "BalancedAccuracy"
                    ].std(),

                "MacroF1Mean":
                    folds["MacroF1"].mean(),

                "MacroF1Std":
                    folds["MacroF1"].std(),

                "Splits":
                    len(folds),
            }
        ]
    )

    # ========================================================
    # GROUP PERFORMANCE
    # ========================================================

    group_rows = []

    for class_name in classes:

        class_predictions = predictions[
            predictions["ActualClass"]
            == class_name
        ]

        n = len(
            class_predictions
        )

        correct = int(
            class_predictions[
                "Correct"
            ].sum()
        )

        recall = (
            correct / n
            if n > 0
            else 0.0
        )

        group_rows.append(
            {
                "Group":
                    class_name,

                "PredictionCount":
                    n,

                "CorrectCount":
                    correct,

                "Recall":
                    recall,
            }
        )

    group_df = pd.DataFrame(
        group_rows
    )

    # ========================================================
    # SAMPLE STABILITY
    # ========================================================

    sample_rows = []

    for sample_id, sample_df in (
        predictions
        .groupby("SampleId")
    ):

        actual_class = (
            sample_df[
                "ActualClass"
            ].iloc[0]
        )

        prediction_counts = (
            sample_df[
                "PredictedClass"
            ]
            .value_counts()
        )

        most_frequent_prediction = (
            prediction_counts.index[0]
        )

        stability = (
            prediction_counts.iloc[0]
            / len(sample_df)
        )

        correct_rate = (
            sample_df[
                "Correct"
            ].mean()
        )

        mean_margin = (
            sample_df[
                "ScoreMargin"
            ].mean()
        )

        sample_rows.append(
            {
                "SampleId":
                    int(sample_id),

                "ActualClass":
                    actual_class,

                "MostFrequentPrediction":
                    most_frequent_prediction,

                "Stability":
                    float(stability),

                "CorrectRate":
                    float(correct_rate),

                "MeanScoreMargin":
                    float(mean_margin),

                "ValidationPredictions":
                    len(sample_df),
            }
        )

    sample_df = (
        pd.DataFrame(
            sample_rows
        )
        .sort_values(
            [
                "CorrectRate",
                "Stability",
            ],
            ascending=[
                False,
                False,
            ],
        )
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

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    group_df.to_csv(
        GROUP_FILE,
        index=False,
    )

    sample_df.to_csv(
        SAMPLE_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print("=" * 72)
    print(
        "CANDIDATE 10 SUMMARY"
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
        "Group performance:"
    )

    print(
        group_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()

    print(
        "Most unstable samples:"
    )

    unstable = (
        sample_df
        .sort_values(
            [
                "Stability",
                "CorrectRate",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .head(10)
    )

    print(
        unstable.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()

    print(
        f"Summary     : {SUMMARY_FILE}"
    )

    print(
        f"Predictions : {PREDICTIONS_FILE}"
    )

    print(
        f"Groups      : {GROUP_FILE}"
    )

    print(
        f"Samples     : {SAMPLE_FILE}"
    )


if __name__ == "__main__":
    main()