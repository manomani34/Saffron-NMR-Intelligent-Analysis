import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42

N_SPLITS = 2
N_REPEATS = 10

TOP_K_VALUES = [10, 20, 30, 50]


# ==========================================================
# Main evaluation function
# ==========================================================

def evaluate_stable_models(
    X_features: pd.DataFrame,
    y: pd.Series,
    results_path: str = "stable_model_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\n" + "=" * 70)
    print("STABLE MODEL EVALUATION")
    print("=" * 70)

    print("\n[1] INPUT DATA")

    print(f"Samples             : {X_features.shape[0]}")
    print(f"Features            : {X_features.shape[1]}")
    print(f"Classes             : {y.nunique()}")
    print(f"CV strategy         : RepeatedStratifiedKFold")
    print(f"Folds per repeat    : {N_SPLITS}")
    print(f"Repeats             : {N_REPEATS}")
    print(f"Total folds         : {N_SPLITS * N_REPEATS}")

    # ------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------

    print("\n[2] CLASS DISTRIBUTION")

    class_counts = (
        y.astype(str)
        .value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda x:
                    int(x[1:])
                    if str(x).startswith("G")
                    else 999
            )
        )
    )

    for group, count in class_counts.items():
        print(f"{group:4} -> {count} samples")

    min_class_count = int(class_counts.min())

    if min_class_count < N_SPLITS:
        raise ValueError(
            f"Smallest class contains only {min_class_count} samples. "
            f"Cannot use {N_SPLITS}-fold stratified CV."
        )

    # ------------------------------------------------------
    # Encode labels
    # ------------------------------------------------------

    print("\n[3] CLASS ENCODING")

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y.astype(str)
    )

    print(
        f"Number of encoded classes : "
        f"{len(label_encoder.classes_)}"
    )

    for index, label in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{label:4} -> {index}"
        )

    # ------------------------------------------------------
    # Repeated cross-validation
    # ------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    models = {

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    all_results = []

    # Silence known sklearn/XGBoost target warnings.
    warning_context = warnings.catch_warnings()
    warning_context.__enter__()

    warnings.filterwarnings(
        "ignore",
        message="The number of unique classes is greater than 50% of the number of samples.*",
    )

    try:

        # ==================================================
        # MODEL LOOP
        # ==================================================

        for model_name, model in models.items():

            print("\n" + "=" * 70)
            print(f"MODEL: {model_name}")
            print("=" * 70)

            for k in TOP_K_VALUES:

                print("\n" + "-" * 70)
                print(
                    f"{model_name} + Top-{k} Features"
                )
                print("-" * 70)

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
                        k,
                        X_train.shape[1],
                    )

                    # --------------------------------------------------
                    # Feature selection must happen INSIDE each fold.
                    # This prevents data leakage.
                    # --------------------------------------------------

                    steps = [
                        (
                            "feature_selection",
                            SelectKBest(
                                score_func=f_classif,
                                k=actual_k,
                            ),
                        )
                    ]

                    # SVM benefits from scaling.
                    if model_name == "SVM":
                        steps.append(
                            (
                                "scaler",
                                StandardScaler(),
                            )
                        )

                    steps.append(
                        (
                            "model",
                            clone(model),
                        )
                    )

                    pipeline = Pipeline(
                        steps=steps
                    )

                    pipeline.fit(
                        X_train,
                        y_train,
                    )

                    y_pred = pipeline.predict(
                        X_test
                    )

                    accuracy = accuracy_score(
                        y_test,
                        y_pred,
                    )

                    balanced_accuracy = (
                        balanced_accuracy_score(
                            y_test,
                            y_pred,
                        )
                    )

                    macro_f1 = f1_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )

                    result = {
                        "Model": model_name,
                        "TopK": actual_k,
                        "Fold": fold_number,
                        "Accuracy": accuracy,
                        "BalancedAccuracy": balanced_accuracy,
                        "F1Macro": macro_f1,
                    }

                    fold_metrics.append(
                        result
                    )

                    all_results.append(
                        result
                    )

                fold_df = pd.DataFrame(
                    fold_metrics
                )

                print(
                    f"Accuracy          : "
                    f"{fold_df['Accuracy'].mean():.4f} "
                    f"+/- "
                    f"{fold_df['Accuracy'].std(ddof=1):.4f}"
                )

                print(
                    f"Balanced Accuracy : "
                    f"{fold_df['BalancedAccuracy'].mean():.4f} "
                    f"+/- "
                    f"{fold_df['BalancedAccuracy'].std(ddof=1):.4f}"
                )

                print(
                    f"Macro F1          : "
                    f"{fold_df['F1Macro'].mean():.4f} "
                    f"+/- "
                    f"{fold_df['F1Macro'].std(ddof=1):.4f}"
                )

    finally:
        warning_context.__exit__(None, None, None)

    # ======================================================
    # Detailed results
    # ======================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================
    # Aggregate results
    # ======================================================

    summary_df = (
        results_df
        .groupby(
            [
                "Model",
                "TopK",
            ]
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

    summary_df = summary_df.fillna(0.0)

    # ------------------------------------------------------
    # Ranking
    # ------------------------------------------------------

    ranked_df = (
        summary_df
        .sort_values(
            by=[
                "BalancedAccuracyMean",
                "F1MacroMean",
                "AccuracyMean",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranked_df.insert(
        0,
        "Rank",
        range(
            1,
            len(ranked_df) + 1,
        ),
    )

    summary_path = results_path.replace(
        ".csv",
        "_summary.csv",
    )

    ranked_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================
    # Terminal output
    # ======================================================

    print("\n" + "=" * 70)
    print("[4] STABLE MODEL RANKING")
    print("=" * 70)

    display_columns = [
        "Rank",
        "Model",
        "TopK",
        "AccuracyMean",
        "BalancedAccuracyMean",
        "F1MacroMean",
    ]

    print(
        ranked_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ======================================================
    # Best candidate
    # ======================================================

    best = ranked_df.iloc[0]

    print("\n" + "=" * 70)
    print("[5] BEST STABLE CANDIDATE")
    print("=" * 70)

    print(
        f"Model             : {best['Model']}"
    )

    print(
        f"TopK              : {int(best['TopK'])}"
    )

    print(
        f"Accuracy          : "
        f"{best['AccuracyMean']:.4f} "
        f"+/- "
        f"{best['AccuracyStd']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{best['BalancedAccuracyMean']:.4f} "
        f"+/- "
        f"{best['BalancedAccuracyStd']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{best['F1MacroMean']:.4f} "
        f"+/- "
        f"{best['F1MacroStd']:.4f}"
    )

    print(
        "\nDetailed results saved to: "
        f"{results_path}"
    )

    print(
        "Summary saved to: "
        f"{summary_path}"
    )

    print("\n" + "=" * 70)
    print("STABLE MODEL EVALUATION FINISHED")
    print("=" * 70)

    return results_df, ranked_df