import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 2


def evaluate_tree_models(
    X_features: pd.DataFrame,
    y: pd.Series,
    results_path: str = "tree_models_results.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("TREE-BASED MODELS")
    print("=" * 70)

    print("\n[1] INPUT DATA")

    print(f"Samples              : {X_features.shape[0]}")
    print(f"Engineered features  : {X_features.shape[1]}")
    print(f"Classes              : {y.nunique()}")

    # --------------------------------------------------
    # 1. Class distribution
    # --------------------------------------------------

    print("\n[2] CLASS DISTRIBUTION")

    class_counts = (
        y.value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda x:
                    int(str(x)[1:])
                    if str(x).startswith("G")
                    else 999
            )
        )
    )

    for group, count in class_counts.items():
        print(f"{group:4} -> {count} samples")

    # --------------------------------------------------
    # 2. Encode target labels
    # --------------------------------------------------
    #
    # XGBoost expects classes as integer labels:
    #
    # G1  -> 0
    # G2  -> 1
    # ...
    # G11 -> 10
    #
    # The encoder is fitted once because the class set
    # is fixed for this dataset.
    # --------------------------------------------------

    print("\n[3] CLASS ENCODING")

    label_encoder = LabelEncoder()

    y_encoded = pd.Series(
        label_encoder.fit_transform(
            y.astype(str)
        ),
        index=y.index,
        name="EncodedClass",
    )

    print(
        f"Number of encoded classes : "
        f"{len(label_encoder.classes_)}"
    )

    print(
        "Mapping:"
    )

    for encoded_value, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{class_name:4} -> {encoded_value}"
        )

    # --------------------------------------------------
    # 3. Cross-validation
    # --------------------------------------------------

    print("\n[4] CROSS-VALIDATION")

    min_class_count = int(
        class_counts.min()
    )

    n_splits = min(
        N_SPLITS,
        min_class_count,
    )

    if n_splits < 2:
        raise ValueError(
            "At least two samples per class are required."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        "Strategy        : StratifiedKFold"
    )

    print(
        f"Folds           : {n_splits}"
    )

    print(
        f"Random state    : {RANDOM_STATE}"
    )

    # --------------------------------------------------
    # 4. Feature-selection values
    # --------------------------------------------------

    feature_counts = [
        10,
        20,
        30,
        50,
    ]

    print("\n[5] FEATURE SELECTION")

    print(
        f"Top-k values    : {feature_counts}"
    )

    # --------------------------------------------------
    # 5. Model factory
    # --------------------------------------------------
    #
    # IMPORTANT:
    # A fresh model is created for every experiment.
    #
    # This prevents fitted state from one experiment
    # being reused in another experiment.
    # --------------------------------------------------

    def create_model(model_name: str):

        if model_name == "RandomForest":

            return RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                max_features="sqrt",
            )

        if model_name == "XGBoost":

            return XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=len(label_encoder.classes_),
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_names = [
        "RandomForest",
        "XGBoost",
    ]

    all_results = []

    # --------------------------------------------------
    # 6. Run models
    # --------------------------------------------------

    for model_name in model_names:

        print("\n" + "=" * 70)
        print(
            f"MODEL: {model_name}"
        )
        print("=" * 70)

        for k in feature_counts:

            print("\n" + "-" * 70)
            print(
                f"{model_name} + Top-{k} Features"
            )
            print("-" * 70)

            fold_results = []

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

                y_train = y_encoded.iloc[
                    train_idx
                ]

                y_test = y_encoded.iloc[
                    test_idx
                ]

                actual_k = min(
                    k,
                    X_train.shape[1],
                )

                # --------------------------------------
                # Fresh model for this fold/experiment
                # --------------------------------------

                model = create_model(
                    model_name
                )

                # --------------------------------------
                # Leakage-safe feature selection
                # --------------------------------------

                pipeline = Pipeline(
                steps=[
            (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=actual_k,
            ),
            ),
            (
            "model",
            model,
            ),
    ]
)

                # --------------------------------------
                # Fit only training fold
                # --------------------------------------

                pipeline.fit(
                    X_train,
                    y_train,
                )

                # --------------------------------------
                # Predict validation fold
                # --------------------------------------

                y_pred_encoded = pipeline.predict(
                    X_test
                )

                # --------------------------------------
                # Convert encoded classes back to G1...G11
                # --------------------------------------

                y_test_original = (
                    label_encoder.inverse_transform(
                        y_test.astype(int)
                    )
                )

                y_pred_original = (
                    label_encoder.inverse_transform(
                        y_pred_encoded.astype(int)
                    )
                )

                # --------------------------------------
                # Metrics
                # --------------------------------------

                accuracy = accuracy_score(
                    y_test_original,
                    y_pred_original,
                )

                balanced_accuracy = (
                    balanced_accuracy_score(
                        y_test_original,
                        y_pred_original,
                    )
                )

                macro_f1 = f1_score(
                    y_test_original,
                    y_pred_original,
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

                fold_results.append(
                    result
                )

                all_results.append(
                    result
                )

                print(
                    f"Fold {fold_number}: "
                    f"Accuracy={accuracy:.4f}, "
                    f"BalancedAccuracy={balanced_accuracy:.4f}, "
                    f"F1Macro={macro_f1:.4f}"
                )

            # ------------------------------------------
            # Experiment summary
            # ------------------------------------------

            fold_df = pd.DataFrame(
                fold_results
            )

            print("\nExperiment summary:")

            print(
                f"Accuracy          : "
                f"{fold_df['Accuracy'].mean():.4f}"
            )

            print(
                f"Balanced Accuracy : "
                f"{fold_df['BalancedAccuracy'].mean():.4f}"
            )

            print(
                f"Macro F1          : "
                f"{fold_df['F1Macro'].mean():.4f}"
            )

    # --------------------------------------------------
    # 7. Final results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 70)
    print("[6] TREE MODEL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 8. Aggregate
    # --------------------------------------------------

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

    print("\n" + "=" * 70)
    print("[7] TREE MODEL SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 9. Save
    # --------------------------------------------------

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = results_path.replace(
        ".csv",
        "_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nDetailed results saved to: "
        f"{results_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("\n" + "=" * 70)
    print("TREE MODELS FINISHED")
    print("=" * 70)

    return summary_df