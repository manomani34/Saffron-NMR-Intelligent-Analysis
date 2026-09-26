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
    N_REPEATS,
    N_SPLITS,
    TOP_K_VALUES,
    build_xgb_model,
    mutual_info_scores,
)


def _build_pipeline(n_classes: int, top_k: int) -> Pipeline:
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
                build_xgb_model(n_classes),
            ),
        ]
    )


def _evaluate_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "F1Macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_robust_models(
    X_features: pd.DataFrame,
    y: pd.Series,
    output_path: str = "robust_evaluation_results.csv",
    summary_path: str = "robust_evaluation_summary.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("ROBUST MODEL EVALUATION")
    print("=" * 70)

    if X_features.empty:
        raise ValueError("Feature matrix is empty.")

    if len(X_features) != len(y):
        raise ValueError("X_features and y row counts do not match.")

    y_string = y.astype(str)

    class_counts = (
        y_string
        .value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda x: int(str(x)[1:])
                if str(x).startswith("G")
                else 999
            )
        )
    )

    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        raise ValueError(
            "At least two samples are required in every class."
        )

    n_splits = min(N_SPLITS, min_class_count)

    print("\n[1] INPUT DATA")
    print(f"Samples             : {len(X_features)}")
    print(f"Features            : {X_features.shape[1]}")
    print(f"Classes             : {y_string.nunique()}")
    print(f"Minimum class size  : {min_class_count}")
    print(f"CV folds            : {n_splits}")
    print(f"CV repeats          : {N_REPEATS}")
    print("Training weighting  : balanced sample weights")
    print("Feature scoring     : mutual information, random_state=42")

    print("\n[2] CLASS DISTRIBUTION")
    for group, count in class_counts.items():
        print(f"{group:4} -> {count} samples")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_string)
    n_classes = len(label_encoder.classes_)

    all_results = []

    print("\n[3] REPEATED STRATIFIED EVALUATION")

    for top_k in TOP_K_VALUES:
        print("\n" + "-" * 70)
        print(f"XGBoost + Top-{top_k}")
        print("-" * 70)

        repeat_metrics = []

        for repeat in range(N_REPEATS):
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE + repeat,
            )

            repeat_predictions = np.full(
                len(y_encoded),
                -1,
                dtype=int,
            )

            actual_k = min(top_k, X_features.shape[1])

            for train_idx, test_idx in cv.split(
                X_features,
                y_encoded,
            ):
                X_train = X_features.iloc[train_idx]
                X_test = X_features.iloc[test_idx]
                y_train = y_encoded[train_idx]

                pipeline = _build_pipeline(
                    n_classes=n_classes,
                    top_k=actual_k,
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

                repeat_predictions[test_idx] = pipeline.predict(
                    X_test
                )

            if np.any(repeat_predictions < 0):
                raise RuntimeError(
                    "Some samples did not receive a robust prediction."
                )

            metrics = _evaluate_predictions(
                y_encoded,
                repeat_predictions,
            )
            metrics["TopK"] = actual_k
            metrics["Repeat"] = repeat + 1

            repeat_metrics.append(metrics)
            all_results.append(metrics)

        repeat_df = pd.DataFrame(repeat_metrics)

        print(
            f"Accuracy          : {repeat_df['Accuracy'].mean():.4f} "
            f"+/- {repeat_df['Accuracy'].std(ddof=1):.4f}"
        )
        print(
            f"Balanced Accuracy : "
            f"{repeat_df['BalancedAccuracy'].mean():.4f} "
            f"+/- {repeat_df['BalancedAccuracy'].std(ddof=1):.4f}"
        )
        print(
            f"Macro F1          : {repeat_df['F1Macro'].mean():.4f} "
            f"+/- {repeat_df['F1Macro'].std(ddof=1):.4f}"
        )

    results_df = pd.DataFrame(all_results)

    summary_df = (
        results_df
        .groupby("TopK")
        .agg(
            AccuracyMean=("Accuracy", "mean"),
            AccuracyStd=("Accuracy", "std"),
            BalancedAccuracyMean=("BalancedAccuracy", "mean"),
            BalancedAccuracyStd=("BalancedAccuracy", "std"),
            F1MacroMean=("F1Macro", "mean"),
            F1MacroStd=("F1Macro", "std"),
        )
        .reset_index()
    )

    n_classes = y_string.nunique()
    chance = 1.0 / n_classes

    summary_df["ChanceAccuracy"] = chance
    summary_df["ChanceBalancedAccuracy"] = chance
    summary_df["AccuracyAboveChance"] = (
        summary_df["AccuracyMean"] - chance
    )
    summary_df["BalancedAccuracyAboveChance"] = (
        summary_df["BalancedAccuracyMean"] - chance
    )
    summary_df["AboveChanceCandidate"] = (
        summary_df["BalancedAccuracyMean"] > chance
    )

    summary_df = (
        summary_df
        .sort_values(
            by=[
                "BalancedAccuracyMean",
                "F1MacroMean",
                "AccuracyMean",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("[4] ROBUST EVALUATION SUMMARY")
    print("=" * 70)
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)
    print("[5] CHANCE-LEVEL REFERENCE")
    print("=" * 70)
    print(f"Random Accuracy             : {chance:.4f}")
    print(f"Random Balanced Accuracy    : {chance:.4f}")

    best = summary_df.iloc[0]

    print("\n" + "=" * 70)
    print("[6] SELECTED ROBUST CONFIGURATION")
    print("=" * 70)
    print(f"Model                  : XGBoost")
    print(f"Top-K                  : {int(best['TopK'])}")
    print(
        f"Accuracy               : {best['AccuracyMean']:.4f} "
        f"+/- {best['AccuracyStd']:.4f}"
    )
    print(
        f"Balanced Accuracy      : "
        f"{best['BalancedAccuracyMean']:.4f} "
        f"+/- {best['BalancedAccuracyStd']:.4f}"
    )
    print(
        f"Macro F1               : {best['F1MacroMean']:.4f} "
        f"+/- {best['F1MacroStd']:.4f}"
    )
    print("Selection criterion     : Balanced Accuracy")
    print("Operational status      : NOT RELIABLE")

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nDetailed results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("ROBUST MODEL EVALUATION FINISHED")
    print("=" * 70)

    return summary_df
