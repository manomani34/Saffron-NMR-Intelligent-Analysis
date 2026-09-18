from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .feature_engineering import (
    build_combined_feature_matrix,
)


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
    / "feature_topk_results.csv"
)

DETAIL_OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "feature_topk_split_results.csv"
)

FEATURE_SELECTION_OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "feature_selection_frequency.csv"
)


# ============================================================
# CONFIG
# ============================================================

TOP_K_VALUES = [
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    108,
]

N_SPLITS = 2
N_REPEATS = 20

N_COMPONENTS = 4

RANDOM_STATE = 42


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
            "SampleId column not found in sample_mapping.csv"
        )

    group_column_candidates = [
        "Group",
        "group",
        "Region",
        "region",
    ]

    group_column = None

    for column in group_column_candidates:

        if column in mapping.columns:
            group_column = column
            break

    if group_column is None:
        raise ValueError(
            "Could not find group column. "
            "Expected one of: "
            "Group, group, Region, region"
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

    X, feature_names = (
        build_combined_feature_matrix(
            data["X_440"],
            data["X_250"],
            data["X_308"],
            data["Time440"],
            data["Time250"],
            data["Time308"],
        )
    )

    return (
        X,
        y,
        classes,
        feature_names,
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
    n_classes: int,
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

    if scores.shape[1] < n_classes:

        padded = np.zeros(
            (
                scores.shape[0],
                n_classes,
            ),
            dtype=float,
        )

        padded[
            :,
            :scores.shape[1],
        ] = scores

        scores = padded

    return np.argmax(
        scores[:, :n_classes],
        axis=1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "HPLC FEATURE ENGINEERING + TOP-K CV"
    )
    print("=" * 72)

    (
        X,
        y,
        classes,
        feature_names,
        sample_ids,
    ) = load_data()

    print(
        f"Independent samples : {len(y)}"
    )

    print(
        f"Classes              : {len(classes)}"
    )

    print(
        f"Features             : {X.shape[1]}"
    )

    print(
        "CV                   : "
        f"Repeated Stratified "
        f"{N_SPLITS}-Fold × {N_REPEATS}"
    )

    print(
        "Top-K values         : "
        + ", ".join(
            map(
                str,
                TOP_K_VALUES,
            )
        )
    )

    print()

    # --------------------------------------------------------
    # Validate feature matrix
    # --------------------------------------------------------

    nan_count = int(
        np.isnan(X).sum()
    )

    inf_count = int(
        np.isinf(X).sum()
    )

    print(
        f"NaN values            : {nan_count}"
    )

    print(
        f"Inf values            : {inf_count}"
    )

    if nan_count > 0:
        raise ValueError(
            "Feature matrix contains NaN."
        )

    if inf_count > 0:
        raise ValueError(
            "Feature matrix contains Inf."
        )

    # --------------------------------------------------------
    # Fixed CV splits
    # --------------------------------------------------------

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

    detailed_rows = []

    feature_selection_rows = []

    n_classes = len(
        classes
    )

    # ========================================================
    # TOP-K LOOP
    # ========================================================

    for k in TOP_K_VALUES:

        if k > X.shape[1]:
            continue

        print(
            f"Top-K = {k}"
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
            # 1. IMPUTATION
            # ------------------------------------------------

            imputer = SimpleImputer(
                strategy="median"
            )

            X_train_imp = (
                imputer.fit_transform(
                    X_train_raw
                )
            )

            X_test_imp = (
                imputer.transform(
                    X_test_raw
                )
            )

            # ------------------------------------------------
            # 2. SCALING
            # FIT ONLY ON TRAIN
            # ------------------------------------------------

            scaler = StandardScaler()

            X_train_scaled = (
                scaler.fit_transform(
                    X_train_imp
                )
            )

            X_test_scaled = (
                scaler.transform(
                    X_test_imp
                )
            )

            # ------------------------------------------------
            # 3. TOP-K SELECTION
            # FIT ONLY ON TRAIN
            # ------------------------------------------------

            f_scores, _ = f_classif(
                X_train_scaled,
                y_train,
            )

            f_scores = np.nan_to_num(
                f_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            order = np.argsort(
                f_scores
            )[::-1]

            selected_idx = (
                order[:k]
            )

            selected_names = [
                feature_names[i]
                for i in selected_idx
            ]

            # Save which features were selected
            for feature_index in selected_idx:

                feature_selection_rows.append(
                    {
                        "TopK": k,
                        "Split": split_id,
                        "FeatureIndex": int(
                            feature_index
                        ),
                        "Feature": feature_names[
                            feature_index
                        ],
                        "FScore": float(
                            f_scores[
                                feature_index
                            ]
                        ),
                    }
                )

            X_train = (
                X_train_scaled[
                    :,
                    selected_idx
                ]
            )

            X_test = (
                X_test_scaled[
                    :,
                    selected_idx
                ]
            )

            # ------------------------------------------------
            # 4. PLS-DA
            # ------------------------------------------------

            y_train_onehot = one_hot(
                y_train,
                n_classes,
            )

            n_components = min(
                N_COMPONENTS,
                X_train.shape[1],
                X_train.shape[0] - 1,
            )

            model = PLSRegression(
                n_components=n_components,
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
                n_classes,
            )

            # ------------------------------------------------
            # 5. METRICS
            # ------------------------------------------------

            accuracy = (
                accuracy_score(
                    y_test,
                    y_pred,
                )
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y_test,
                    y_pred,
                )
            )

            macro_f1 = (
                f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            )

            detailed_rows.append(
                {
                    "TopK": k,
                    "Split": split_id,
                    "TrainSize": len(
                        train_idx
                    ),
                    "TestSize": len(
                        test_idx
                    ),
                    "Accuracy": accuracy,
                    "BalancedAccuracy":
                        balanced_accuracy,
                    "MacroF1": macro_f1,
                }
            )

        print(
            f"    completed "
            f"{len(splits)} splits"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    detail = pd.DataFrame(
        detailed_rows
    )

    summary = (
        detail
        .groupby("TopK")
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

    summary["FeatureCount"] = (
        summary["TopK"]
    )

    summary = summary[
        [
            "TopK",
            "FeatureCount",
            "AccuracyMean",
            "AccuracyStd",
            "BalancedAccuracyMean",
            "BalancedAccuracyStd",
            "MacroF1Mean",
            "MacroF1Std",
            "Splits",
        ]
    ]

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

    detail.to_csv(
        DETAIL_OUTPUT_FILE,
        index=False,
    )

    selection_df = pd.DataFrame(
        feature_selection_rows
    )

    if not selection_df.empty:

        selection_frequency = (
            selection_df
            .groupby(
                [
                    "TopK",
                    "Feature",
                ]
            )
            .agg(
                SelectionCount=(
                    "Split",
                    "count",
                ),
                MeanFScore=(
                    "FScore",
                    "mean",
                ),
            )
            .reset_index()
        )

        selection_frequency[
            "SelectionRate"
        ] = (
            selection_frequency[
                "SelectionCount"
            ]
            / len(splits)
        )

        selection_frequency = (
            selection_frequency
            .sort_values(
                [
                    "TopK",
                    "SelectionRate",
                    "MeanFScore",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
        )

        selection_frequency.to_csv(
            FEATURE_SELECTION_OUTPUT_FILE,
            index=False,
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print("=" * 72)
    print(
        "TOP-K SUMMARY"
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
        f"Summary output : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Detail output  : "
        f"{DETAIL_OUTPUT_FILE}"
    )

    print(
        f"Feature freq.  : "
        f"{FEATURE_SELECTION_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()