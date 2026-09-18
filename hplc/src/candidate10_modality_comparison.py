from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ASLS_ITERATIONS,
    ASLS_LAMBDA,
    ASLS_P,
    CV_N_SPLITS,
    CV_REPEATS,
    HPLC_RAW_PATH,
    MAPPING_PATH,
    RANDOM_STATE,
    SHEETS,
    OUTPUTS_DIR,
)
from .data_loader import load_hplc_data
from .preprocessing import apply_preprocessing


# ============================================================
# SETTINGS
# ============================================================

PLS_COMPONENTS = 10

PREPROCESSING_METHOD = "asls_snv"

MODALITIES = [
    "250 nm",
    "308 nm",
    "440 nm",
    "Combined",
]

PROBLEM_SAMPLE_IDS = [
    1,
    3,
    8,
    11,
    12,
    18,
    20,
    31,
    38,
    42,
]


# ============================================================
# OUTPUTS
# ============================================================

DETAILED_OUTPUT = (
    OUTPUTS_DIR
    / "candidate10_modality_comparison_predictions.csv"
)

SAMPLE_OUTPUT = (
    OUTPUTS_DIR
    / "candidate10_modality_comparison_samples.csv"
)

SUMMARY_OUTPUT = (
    OUTPUTS_DIR
    / "candidate10_modality_comparison_summary.csv"
)


# ============================================================
# MODEL
# ============================================================

def build_candidate10_pipeline() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "plsda",
                PLSRegression(
                    n_components=PLS_COMPONENTS,
                    scale=False,
                    max_iter=2000,
                    tol=1e-6,
                ),
            ),
        ]
    )


# ============================================================
# CLASS ENCODING
# ============================================================

def one_hot(
    y: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:

    return (
        y[:, None]
        == classes[None, :]
    ).astype(float)


# ============================================================
# PREDICTION
# ============================================================

def predict_classes(
    model: Pipeline,
    X: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    scores = np.asarray(
        model.predict(X),
        dtype=float,
    )

    winner_idx = np.argmax(
        scores,
        axis=1,
    )

    pred = classes[
        winner_idx
    ]

    if scores.shape[1] >= 2:

        ordered = np.sort(
            scores,
            axis=1,
        )

        margins = (
            ordered[:, -1]
            - ordered[:, -2]
        )

    else:

        margins = np.abs(
            scores[:, 0]
        )

    return pred, margins


# ============================================================
# LOAD DATA
# ============================================================

def load_independent_data():

    print()
    print(
        "[1] Loading authoritative HPLC data..."
    )

    data = load_hplc_data(
        HPLC_RAW_PATH,
        MAPPING_PATH,
    )

    independent = (
        data.mapping[
            data.mapping["Independent"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    sample_ids = (
        independent[
            "SampleId"
        ]
        .astype(int)
        .to_numpy()
    )

    y = (
        independent[
            "Group"
        ]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    positions = {
        code: {
            int(sample_id): idx
            for idx, sample_id
            in enumerate(
                item.sample_ids.tolist()
            )
        }
        for code, item
        in data.wavelengths.items()
    }

    X_by_modality = {}

    for code, label in [
        ("440", "440 nm"),
        ("250", "250 nm"),
        ("308", "308 nm"),
    ]:

        item = data.wavelengths[
            code
        ]

        indices = [
            positions[code][
                sample_id
            ]
            for sample_id
            in sample_ids
        ]

        X_by_modality[label] = (
            np.asarray(
                item.values[
                    indices
                ],
                dtype=float,
            )
        )

    return (
        data,
        independent,
        sample_ids,
        y,
        X_by_modality,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_modalities(
    X_by_modality,
):

    print()
    print(
        "[2] Preprocessing..."
    )

    prepared = {}

    for modality, X in (
        X_by_modality.items()
    ):

        print(
            f"  {modality}: "
            f"AsLS + SNV"
        )

        prepared[modality] = (
            apply_preprocessing(
                X,
                method=PREPROCESSING_METHOD,
                asls_lambda=ASLS_LAMBDA,
                asls_p=ASLS_P,
                asls_iterations=ASLS_ITERATIONS,
            )
        )

    prepared["Combined"] = (
        np.hstack(
            [
                prepared["440 nm"],
                prepared["250 nm"],
                prepared["308 nm"],
            ]
        )
    )

    return prepared


# ============================================================
# CV
# ============================================================

def build_splits(
    X: np.ndarray,
    y: np.ndarray,
):

    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    return list(
        cv.split(
            X,
            y,
        )
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_modality(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    classes: np.ndarray,
    Y: np.ndarray,
    splits,
    modality: str,
):

    prediction_rows = []
    metric_rows = []

    for split_id, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits,
        start=1,
    ):

        repeat = (
            (split_id - 1)
            // CV_N_SPLITS
        ) + 1

        fold = (
            (split_id - 1)
            % CV_N_SPLITS
        ) + 1

        model = (
            build_candidate10_pipeline()
        )

        model.fit(
            X[train_idx],
            Y[train_idx],
        )

        pred, margins = (
            predict_classes(
                model,
                X[test_idx],
                classes,
            )
        )

        accuracy = accuracy_score(
            y[test_idx],
            pred,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y[test_idx],
                pred,
            )
        )

        macro_f1 = f1_score(
            y[test_idx],
            pred,
            average="macro",
            zero_division=0,
        )

        metric_rows.append(
            {
                "Model": "PLS-DA",
                "PLSComponents":
                    PLS_COMPONENTS,
                "Preprocessing":
                    "AsLS + SNV",
                "Modality":
                    modality,
                "Split":
                    split_id,
                "Repeat":
                    repeat,
                "Fold":
                    fold,
                "Accuracy":
                    accuracy,
                "BalancedAccuracy":
                    balanced_accuracy,
                "MacroF1":
                    macro_f1,
            }
        )

        for local_pos, idx in enumerate(
            test_idx
        ):

            prediction_rows.append(
                {
                    "Model":
                        "PLS-DA",

                    "PLSComponents":
                        PLS_COMPONENTS,

                    "Preprocessing":
                        "AsLS + SNV",

                    "Modality":
                        modality,

                    "Split":
                        split_id,

                    "Repeat":
                        repeat,

                    "Fold":
                        fold,

                    "SampleId":
                        int(
                            sample_ids[idx]
                        ),

                    "ActualGroup":
                        y[idx],

                    "PredictedGroup":
                        pred[local_pos],

                    "Correct":
                        bool(
                            pred[local_pos]
                            == y[idx]
                        ),

                    "DecisionMargin":
                        float(
                            margins[local_pos]
                        ),
                }
            )

    return (
        pd.DataFrame(
            prediction_rows
        ),
        pd.DataFrame(
            metric_rows
        ),
    )


# ============================================================
# DOMINANT PREDICTION PER SAMPLE
# ============================================================

def summarize_samples(
    predictions,
):

    rows = []

    for (
        modality,
        group_modality,
    ) in predictions.groupby(
        "Modality"
    ):

        for sample_id, group in (
            group_modality.groupby(
                "SampleId"
            )
        ):

            actual = (
                group[
                    "ActualGroup"
                ].iloc[0]
            )

            counts = (
                group[
                    "PredictedGroup"
                ]
                .value_counts()
            )

            dominant = (
                counts.index[0]
            )

            dominant_count = (
                counts.iloc[0]
            )

            prediction_count = (
                len(group)
            )

            stability = (
                dominant_count
                / prediction_count
            )

            correct_rate = (
                group["Correct"]
                .mean()
            )

            rows.append(
                {
                    "SampleId":
                        int(sample_id),

                    "ActualClass":
                        actual,

                    "Modality":
                        modality,

                    "DominantPrediction":
                        dominant,

                    "Stability":
                        float(
                            stability
                        ),

                    "CorrectRate":
                        float(
                            correct_rate
                        ),

                    "PredictionCount":
                        int(
                            prediction_count
                        ),

                    "IsCorrect":
                        bool(
                            dominant
                            == actual
                        ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def summarize_metrics(
    metrics,
):

    if metrics.empty:
        return pd.DataFrame()

    summary = (
        metrics.groupby(
            [
                "Model",
                "PLSComponents",
                "Preprocessing",
                "Modality",
            ],
            as_index=False,
        )
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
            TotalSplits=(
                "Split",
                "count",
            ),
        )
    )

    summary[
        "ChanceBalancedAccuracy"
    ] = 1.0 / 11.0

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "CANDIDATE 10 MODALITY COMPARISON"
    )
    print(
        "=" * 78
    )

    print()
    print(
        f"PLS components : {PLS_COMPONENTS}"
    )

    print(
        f"Preprocessing  : AsLS + SNV"
    )

    print(
        f"CV             : "
        f"Repeated Stratified "
        f"{CV_N_SPLITS}-Fold × "
        f"{CV_REPEATS}"
    )

    print(
        f"Random state   : "
        f"{RANDOM_STATE}"
    )

    print(
        f"Total splits   : "
        f"{CV_N_SPLITS * CV_REPEATS}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    (
        data,
        independent,
        sample_ids,
        y,
        X_by_modality,
    ) = load_independent_data()

    classes = np.array(
        sorted(
            np.unique(y)
        ),
        dtype=str,
    )

    Y = one_hot(
        y,
        classes,
    )

    print()
    print(
        f"Independent samples : "
        f"{len(sample_ids)}"
    )

    print(
        f"Classes             : "
        f"{len(classes)}"
    )

    print(
        "Class distribution:"
    )

    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # PREPROCESS
    # ========================================================

    prepared = (
        prepare_modalities(
            X_by_modality
        )
    )

    # ========================================================
    # SAME FIXED SPLITS FOR ALL MODALITIES
    # ========================================================

    print()
    print(
        "[3] Building fixed CV splits..."
    )

    splits = build_splits(
        prepared["440 nm"],
        y,
    )

    print(
        f"  Splits generated: "
        f"{len(splits)}"
    )

    # ========================================================
    # EVALUATE ALL FOUR MODALITIES
    # ========================================================

    all_predictions = []
    all_metrics = []

    print()
    print(
        "[4] Evaluating modalities..."
    )

    for modality in MODALITIES:

        print()
        print(
            "-" * 78
        )

        print(
            f"MODALITY: {modality}"
        )

        X = prepared[
            modality
        ]

        print(
            f"Variables: {X.shape[1]}"
        )

        predictions, metrics = (
            evaluate_modality(
                X=X,
                y=y,
                sample_ids=sample_ids,
                classes=classes,
                Y=Y,
                splits=splits,
                modality=modality,
            )
        )

        all_predictions.append(
            predictions
        )

        all_metrics.append(
            metrics
        )

        print(
            f"Accuracy: "
            f"{metrics['Accuracy'].mean():.4f} "
            f"+/- "
            f"{metrics['Accuracy'].std():.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['BalancedAccuracy'].mean():.4f} "
            f"+/- "
            f"{metrics['BalancedAccuracy'].std():.4f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['MacroF1'].mean():.4f} "
            f"+/- "
            f"{metrics['MacroF1'].std():.4f}"
        )

    # ========================================================
    # COMBINE OUTPUTS
    # ========================================================

    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metric_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    # ========================================================
    # PER SAMPLE SUMMARY
    # ========================================================

    sample_df = (
        summarize_samples(
            prediction_df
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = (
        summarize_metrics(
            metric_df
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    prediction_df.to_csv(
        DETAILED_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    sample_df.to_csv(
        SAMPLE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "FULL MODALITY SUMMARY"
    )
    print(
        "=" * 78
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # PROBLEM SAMPLES ONLY
    # ========================================================

    problem_df = sample_df[
        sample_df["SampleId"].isin(
            PROBLEM_SAMPLE_IDS
        )
    ].copy()

    problem_df[
        "_order"
    ] = problem_df[
        "SampleId"
    ].apply(
        lambda x:
            (
                PROBLEM_SAMPLE_IDS.index(x)
                if x in PROBLEM_SAMPLE_IDS
                else 999
            )
    )

    problem_df = (
        problem_df
        .sort_values(
            [
                "_order",
                "Modality",
            ]
        )
        .drop(
            columns="_order"
        )
    )

    print()
    print(
        "=" * 78
    )
    print(
        "PROBLEM SAMPLES — 250 / 308 / 440 / COMBINED"
    )
    print(
        "=" * 78
    )

    print(
        problem_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # COMPACT SAMPLE MATRIX
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "SAMPLE × MODALITY"
    )
    print(
        "=" * 78
    )

    for sample_id in (
        PROBLEM_SAMPLE_IDS
    ):

        sample_rows = (
            problem_df[
                problem_df["SampleId"]
                == sample_id
            ]
        )

        if sample_rows.empty:
            continue

        actual = (
            sample_rows[
                "ActualClass"
            ].iloc[0]
        )

        print()
        print(
            f"Sample {sample_id} "
            f"(Actual={actual})"
        )

        for modality in MODALITIES:

            row = sample_rows[
                sample_rows[
                    "Modality"
                ]
                == modality
            ]

            if row.empty:

                print(
                    f"  {modality:9s}: MISSING"
                )

                continue

            r = row.iloc[0]

            status = (
                "OK"
                if bool(r["IsCorrect"])
                else "WRONG"
            )

            print(
                f"  {modality:9s}: "
                f"{r['DominantPrediction']} "
                f"| stability="
                f"{r['Stability']:.2f} "
                f"| correct="
                f"{r['CorrectRate']:.2f} "
                f"| {status}"
            )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "OUTPUTS"
    )
    print(
        "=" * 78
    )

    print(
        f"Predictions : "
        f"{DETAILED_OUTPUT}"
    )

    print(
        f"Samples     : "
        f"{SAMPLE_OUTPUT}"
    )

    print(
        f"Summary     : "
        f"{SUMMARY_OUTPUT}"
    )

    print()


if __name__ == "__main__":
    main()