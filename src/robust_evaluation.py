import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 2
N_REPEATS = 20
TOP_K_VALUES = [10, 20, 30, 50]


def _build_model(
    n_classes: int,
) -> XGBClassifier:

    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


def _evaluate_predictions(
    y_true,
    y_pred,
):
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
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
        raise ValueError(
            "Feature matrix is empty."
        )

    if len(X_features) != len(y):
        raise ValueError(
            "X_features and y row counts do not match."
        )

    y_string = y.astype(str)

    class_counts = (
        y_string
        .value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda x:
                    int(str(x)[1:])
                    if str(x).startswith("G")
                    else 999
            )
        )
    )

    min_class_count = int(
        class_counts.min()
    )

    if min_class_count < 2:
        raise ValueError(
            "At least two samples are required "
            "in every class."
        )

    n_splits = min(
        N_SPLITS,
        min_class_count,
    )

    print("\n[1] INPUT DATA")

    print(
        f"Samples             : "
        f"{len(X_features)}"
    )

    print(
        f"Features            : "
        f"{X_features.shape[1]}"
    )

    print(
        f"Classes             : "
        f"{y_string.nunique()}"
    )

    print(
        f"Minimum class size  : "
        f"{min_class_count}"
    )

    print(
        f"CV folds            : "
        f"{n_splits}"
    )

    print(
        f"CV repeats          : "
        f"{N_REPEATS}"
    )

    print("\n[2] CLASS DISTRIBUTION")

    for group, count in class_counts.items():
        print(
            f"{group:4} -> {count} samples"
        )

    label_encoder = LabelEncoder()

    y_encoded = (
        label_encoder.fit_transform(
            y_string
        )
    )

    all_results = []

    print(
        "\n[3] REPEATED STRATIFIED EVALUATION"
    )

    for top_k in TOP_K_VALUES:

        print(
            "\n" + "-" * 70
        )

        print(
            f"XGBoost + Top-{top_k}"
        )

        print(
            "-" * 70
        )

        repeat_metrics = []

        for repeat in range(
            N_REPEATS
        ):

            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=(
                    RANDOM_STATE + repeat
                ),
            )

            actual_k = min(
                top_k,
                X_features.shape[1],
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "feature_selection",
                        SelectKBest(
                            score_func=(
                                mutual_info_classif
                            ),
                            k=actual_k,
                        ),
                    ),
                    (
                        "model",
                        _build_model(
                            len(
                                label_encoder.classes_
                            )
                        ),
                    ),
                ]
            )

            predictions = cross_val_predict(
                pipeline,
                X_features,
                y_encoded,
                cv=cv,
                method="predict",
                n_jobs=1,
            )

            metrics = _evaluate_predictions(
                y_encoded,
                predictions,
            )

            metrics["TopK"] = actual_k
            metrics["Repeat"] = (
                repeat + 1
            )

            repeat_metrics.append(
                metrics
            )

            all_results.append(
                metrics
            )

        repeat_df = pd.DataFrame(
            repeat_metrics
        )

        print(
            f"Accuracy          : "
            f"{repeat_df['Accuracy'].mean():.4f} "
            f"+/- "
            f"{repeat_df['Accuracy'].std(ddof=1):.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{repeat_df['BalancedAccuracy'].mean():.4f} "
            f"+/- "
            f"{repeat_df['BalancedAccuracy'].std(ddof=1):.4f}"
        )

        print(
            f"Macro F1          : "
            f"{repeat_df['F1Macro'].mean():.4f} "
            f"+/- "
            f"{repeat_df['F1Macro'].std(ddof=1):.4f}"
        )

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = (
        results_df
        .groupby(
            "TopK"
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
            F1MacroMean=(
                "F1Macro",
                "mean",
            ),
            F1MacroStd=(
                "F1Macro",
                "std",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Chance-level reference
    # --------------------------------------------------

    n_classes = y_string.nunique()

    chance_accuracy = (
        1.0 / n_classes
    )

    chance_balanced_accuracy = (
        1.0 / n_classes
    )

    summary_df[
        "ChanceAccuracy"
    ] = chance_accuracy

    summary_df[
        "ChanceBalancedAccuracy"
    ] = chance_balanced_accuracy

    summary_df[
        "AccuracyAboveChance"
    ] = (
        summary_df["AccuracyMean"]
        - chance_accuracy
    )

    summary_df[
        "BalancedAccuracyAboveChance"
    ] = (
        summary_df["BalancedAccuracyMean"]
        - chance_balanced_accuracy
    )

    # This means only "above chance".
    # It does NOT mean reliable or operationally valid.
    summary_df[
        "AboveChanceCandidate"
    ] = (
        summary_df[
            "BalancedAccuracyMean"
        ]
        > chance_balanced_accuracy
    )

    summary_df = (
        summary_df.sort_values(
            by=[
                "BalancedAccuracyMean",
                "F1MacroMean",
                "AccuracyMean",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "[4] ROBUST EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}"
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "[5] CHANCE-LEVEL REFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        f"Random Accuracy             : "
        f"{chance_accuracy:.4f}"
    )

    print(
        f"Random Balanced Accuracy    : "
        f"{chance_balanced_accuracy:.4f}"
    )

    best = summary_df.iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "[6] BEST ROBUST CANDIDATE"
    )

    print(
        "=" * 70
    )

    print(
        f"Top-K                : "
        f"{int(best['TopK'])}"
    )

    print(
        f"Accuracy             : "
        f"{best['AccuracyMean']:.4f} "
        f"+/- {best['AccuracyStd']:.4f}"
    )

    print(
        f"Balanced Accuracy    : "
        f"{best['BalancedAccuracyMean']:.4f} "
        f"+/- {best['BalancedAccuracyStd']:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{best['F1MacroMean']:.4f} "
        f"+/- {best['F1MacroStd']:.4f}"
    )

    print(
        f"Above chance (BA)    : "
        f"{best['BalancedAccuracyAboveChance']:.4f}"
    )

    if bool(
        best["AboveChanceCandidate"]
    ):

        print(
            "Candidate status     : "
            "ABOVE CHANCE ONLY"
        )

        print(
            "Operational status   : "
            "NOT RELIABLE"
        )

    else:

        print(
            "Candidate status     : "
            "NOT ABOVE CHANCE"
        )

        print(
            "Operational status   : "
            "NOT RELIABLE"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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

    print(
        f"\nDetailed results saved to: "
        f"{output_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ROBUST MODEL EVALUATION FINISHED"
    )

    print(
        "=" * 70
    )

    return summary_df