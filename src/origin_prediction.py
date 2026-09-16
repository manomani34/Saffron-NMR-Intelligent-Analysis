# src/origin_prediction.py

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 2

# Best configuration observed in the current stable evaluation
TOP_K = 10


def _create_pipeline(
    class_count: int,
    top_k: int,
) -> Pipeline:

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            (
                "feature_selection",
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=top_k,
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def run_origin_prediction(
    X_features: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    output_path: str = "sample_predictions.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("ORIGIN PREDICTION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Validation
    # --------------------------------------------------

    if X_features.shape[0] != len(y):
        raise ValueError(
            "X_features and y row counts do not match."
        )

    if metadata.shape[0] != len(y):
        raise ValueError(
            "Metadata and y row counts do not match."
        )

    if X_features.empty:
        raise ValueError(
            "Feature matrix is empty."
        )

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    # --------------------------------------------------
    # 2. Encode target
    # --------------------------------------------------

    print("\n[1] TARGET ENCODING")

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y.astype(str)
    )

    class_count = len(
        label_encoder.classes_
    )

    if class_count < 2:
        raise ValueError(
            "At least two origin classes are required."
        )

    print(
        f"Samples             : {len(y)}"
    )

    print(
        f"Classes             : {class_count}"
    )

    print(
        f"Top-K features      : {TOP_K}"
    )

    for encoded_value, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{class_name:4} -> {encoded_value}"
        )

    # --------------------------------------------------
    # 3. Class distribution
    # --------------------------------------------------

    class_counts = (
        y.astype(str)
        .value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda value:
                    int(str(value)[1:])
                    if str(value).startswith("G")
                    else 999
            )
        )
    )

    print("\n[2] CLASS DISTRIBUTION")

    for group, count in class_counts.items():
        print(
            f"{group:4} -> {count} samples"
        )

    min_class_count = int(
        class_counts.min()
    )

    n_splits = min(
        N_SPLITS,
        min_class_count,
    )

    if n_splits < 2:
        raise ValueError(
            "At least two samples are required "
            "for every origin class."
        )

    # --------------------------------------------------
    # 4. Cross-validation
    # --------------------------------------------------

    print("\n[3] OUT-OF-FOLD EVALUATION")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_predictions = np.full(
        len(y),
        -1,
        dtype=int,
    )

    oof_probabilities = np.zeros(
        (
            len(y),
            class_count,
        ),
        dtype=float,
    )

    fold_metrics = []

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X_features,
            y_encoded,
        ),
        start=1,
    ):

        print(
            f"\nFold {fold_number}"
        )

        X_train = X_features.iloc[
            train_idx
        ]

        X_test = X_features.iloc[
            test_idx
        ]

        y_train = y_encoded[
            train_idx
        ]

        y_test = y_encoded[
            test_idx
        ]

        actual_k = min(
            TOP_K,
            X_train.shape[1],
        )

        pipeline = _create_pipeline(
            class_count=class_count,
            top_k=actual_k,
        )

        # Balanced weighting for training fold
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights,
        )

        fold_prediction = pipeline.predict(
            X_test
        )

        fold_probability = (
            pipeline.predict_proba(
                X_test
            )
        )

        oof_predictions[
            test_idx
        ] = fold_prediction

        model_classes = (
            pipeline.named_steps[
                "model"
            ].classes_
        )

        for local_index, class_value in enumerate(
            model_classes
        ):

            oof_probabilities[
                test_idx,
                int(class_value),
            ] = fold_probability[
                :,
                local_index,
            ]

        fold_accuracy = accuracy_score(
            y_test,
            fold_prediction,
        )

        fold_balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                fold_prediction,
            )
        )

        fold_f1 = f1_score(
            y_test,
            fold_prediction,
            average="macro",
            zero_division=0,
        )

        fold_metrics.append(
            {
                "Fold": fold_number,
                "Accuracy": fold_accuracy,
                "BalancedAccuracy": (
                    fold_balanced_accuracy
                ),
                "F1Macro": fold_f1,
            }
        )

        print(
            f"Accuracy          : "
            f"{fold_accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{fold_balanced_accuracy:.4f}"
        )

        print(
            f"Macro F1          : "
            f"{fold_f1:.4f}"
        )

    # --------------------------------------------------
    # 5. OOF metrics
    # --------------------------------------------------

    if np.any(
        oof_predictions < 0
    ):
        raise RuntimeError(
            "Some samples did not receive "
            "an out-of-fold prediction."
        )

    oof_predicted_groups = (
        label_encoder.inverse_transform(
            oof_predictions
        )
    )

    actual_groups = (
        y.astype(str)
        .to_numpy()
    )

    oof_confidence = (
        oof_probabilities.max(
            axis=1
        )
    )

    oof_accuracy = accuracy_score(
        actual_groups,
        oof_predicted_groups,
    )

    oof_balanced_accuracy = (
        balanced_accuracy_score(
            actual_groups,
            oof_predicted_groups,
        )
    )

    oof_macro_f1 = f1_score(
        actual_groups,
        oof_predicted_groups,
        average="macro",
        zero_division=0,
    )

    print("\n" + "-" * 70)
    print("OOF SUMMARY")
    print("-" * 70)

    print(
        f"Accuracy          : "
        f"{oof_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{oof_balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{oof_macro_f1:.4f}"
    )

    # --------------------------------------------------
    # 6. Final model on all available reference data
    # --------------------------------------------------

    print("\n[4] FINAL REFERENCE MODEL")

    final_k = min(
        TOP_K,
        X_features.shape[1],
    )

    final_pipeline = _create_pipeline(
        class_count=class_count,
        top_k=final_k,
    )

    final_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_encoded,
    )

    final_pipeline.fit(
        X_features,
        y_encoded,
        model__sample_weight=final_sample_weights,
    )

    final_predictions = (
        final_pipeline.predict(
            X_features
        )
    )

    final_probabilities = (
        final_pipeline.predict_proba(
            X_features
        )
    )

    final_predicted_groups = (
        label_encoder.inverse_transform(
            final_predictions
        )
    )

    final_confidence = (
        final_probabilities.max(
            axis=1
        )
    )

    # --------------------------------------------------
    # 7. Build result
    # --------------------------------------------------

    print("\n[5] SAVING OOF PREDICTIONS")

    if "SampleId" in metadata.columns:
        sample_ids = metadata[
            "SampleId"
        ].to_numpy()
    else:
        sample_ids = np.arange(
            1,
            len(y) + 1,
        )

    results = pd.DataFrame(
        {
            "SampleId": sample_ids,
            "ActualGroup": actual_groups,
            "PredictedGroup": (
                oof_predicted_groups
            ),
            "PredictionConfidence": (
                oof_confidence
            ),
            "Correct": (
                actual_groups
                == oof_predicted_groups
            ),
            "FinalModelPrediction": (
                final_predicted_groups
            ),
            "FinalModelConfidence": (
                final_confidence
            ),
        }
    )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print("\n[6] PREDICTION SUMMARY")

    print(
        results[
            [
                "SampleId",
                "ActualGroup",
                "PredictedGroup",
                "PredictionConfidence",
                "Correct",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print("\n[7] FOLD METRICS")

    fold_df = pd.DataFrame(
        fold_metrics
    )

    print(
        fold_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("ORIGIN PREDICTION FINISHED")
    print("=" * 70)

    return results