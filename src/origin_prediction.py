import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.origin_model_config import (
    RANDOM_STATE,
    N_SPLITS,
    build_xgb_model,
    mutual_info_scores,
)


def _create_pipeline(class_count: int, top_k: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_selection",
                SelectKBest(
                    score_func=mutual_info_scores,
                    k=top_k,
                ),
            ),
            (
                "model",
                build_xgb_model(class_count),
            ),
        ]
    )


def run_origin_prediction(
    X_features: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    top_k: int,
    output_path: str = "sample_predictions.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("ORIGIN PREDICTION")
    print("=" * 70)

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if X_features.shape[0] != len(y):
        raise ValueError("X_features and y row counts do not match.")

    if metadata.shape[0] != len(y):
        raise ValueError("Metadata and y row counts do not match.")

    if X_features.empty:
        raise ValueError("Feature matrix is empty.")

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    class_count = len(label_encoder.classes_)

    if class_count < 2:
        raise ValueError("At least two origin classes are required.")

    actual_top_k = min(top_k, X_features.shape[1])

    print("\n[1] TARGET ENCODING")
    print(f"Samples             : {len(y)}")
    print(f"Classes             : {class_count}")
    print(f"Selected Top-K      : {actual_top_k}")
    print("Configuration source: Robust Evaluation")

    for encoded_value, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name:4} -> {encoded_value}")

    class_counts = (
        y.astype(str)
        .value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda value: int(str(value)[1:])
                if str(value).startswith("G")
                else 999
            )
        )
    )

    print("\n[2] CLASS DISTRIBUTION")
    for group, count in class_counts.items():
        print(f"{group:4} -> {count} samples")

    min_class_count = int(class_counts.min())
    n_splits = min(N_SPLITS, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "At least two samples are required for every origin class."
        )

    print("\n[3] OUT-OF-FOLD EVALUATION")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_predictions = np.full(len(y), -1, dtype=int)
    oof_probabilities = np.zeros(
        (len(y), class_count),
        dtype=float,
    )
    fold_metrics = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X_features, y_encoded),
        start=1,
    ):
        print(f"\nFold {fold_number}")

        X_train = X_features.iloc[train_idx]
        X_test = X_features.iloc[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        pipeline = _create_pipeline(
            class_count=class_count,
            top_k=actual_top_k,
        )

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights,
        )

        fold_prediction = pipeline.predict(X_test)
        fold_probability = pipeline.predict_proba(X_test)

        oof_predictions[test_idx] = fold_prediction

        model_classes = pipeline.named_steps["model"].classes_
        for local_index, class_value in enumerate(model_classes):
            oof_probabilities[
                test_idx,
                int(class_value),
            ] = fold_probability[:, local_index]

        fold_accuracy = accuracy_score(y_test, fold_prediction)
        fold_balanced_accuracy = balanced_accuracy_score(
            y_test,
            fold_prediction,
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
                "BalancedAccuracy": fold_balanced_accuracy,
                "F1Macro": fold_f1,
            }
        )

        print(f"Accuracy          : {fold_accuracy:.4f}")
        print(f"Balanced Accuracy : {fold_balanced_accuracy:.4f}")
        print(f"Macro F1          : {fold_f1:.4f}")

    if np.any(oof_predictions < 0):
        raise RuntimeError(
            "Some samples did not receive an out-of-fold prediction."
        )

    oof_predicted_groups = label_encoder.inverse_transform(oof_predictions)
    actual_groups = y.astype(str).to_numpy()
    oof_confidence = oof_probabilities.max(axis=1)

    oof_accuracy = accuracy_score(actual_groups, oof_predicted_groups)
    oof_balanced_accuracy = balanced_accuracy_score(
        actual_groups,
        oof_predicted_groups,
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
    print(f"Accuracy          : {oof_accuracy:.4f}")
    print(f"Balanced Accuracy : {oof_balanced_accuracy:.4f}")
    print(f"Macro F1          : {oof_macro_f1:.4f}")

    print("\n[4] FINAL REFERENCE MODEL")

    final_pipeline = _create_pipeline(
        class_count=class_count,
        top_k=actual_top_k,
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

    final_predictions = final_pipeline.predict(X_features)
    final_probabilities = final_pipeline.predict_proba(X_features)
    final_predicted_groups = label_encoder.inverse_transform(
        final_predictions
    )
    final_confidence = final_probabilities.max(axis=1)

    print("\n[5] SAVING OOF PREDICTIONS")

    if "SampleId" in metadata.columns:
        sample_ids = metadata["SampleId"].to_numpy()
    else:
        sample_ids = np.arange(1, len(y) + 1)

    results = pd.DataFrame(
        {
            "SampleId": sample_ids,
            "ActualGroup": actual_groups,
            "PredictedGroup": oof_predicted_groups,
            "PredictionConfidence": oof_confidence,
            "Correct": actual_groups == oof_predicted_groups,
            "FinalModelPrediction": final_predicted_groups,
            "FinalModelConfidence": final_confidence,
            "Model": "XGBoost",
            "TopK": actual_top_k,
        }
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Results saved to: {output_path}")

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
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n[7] FOLD METRICS")
    fold_df = pd.DataFrame(fold_metrics)
    print(
        fold_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nConfiguration consistency:")
    print(f"  Robust-selected Top-K : {actual_top_k}")
    print(f"  OOF Top-K             : {actual_top_k}")
    print("  Feature selector      : mutual information")
    print("  Sample weighting      : balanced")
    print("  XGBoost configuration : shared")

    print("\n" + "=" * 70)
    print("ORIGIN PREDICTION FINISHED")
    print("=" * 70)

    return results
