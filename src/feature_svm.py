import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
N_SPLITS = 2


def evaluate_feature_svm(
    X_features: pd.DataFrame,
    y: pd.Series,
    results_path: str = "feature_svm_results.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURES + SVM")
    print("=" * 70)

    print("\n[1] INPUT DATA")

    print(
        f"Samples         : {X_features.shape[0]}"
    )

    print(
        f"Engineered features : {X_features.shape[1]}"
    )

    print(
        f"Classes         : {y.nunique()}"
    )

    # --------------------------------------------------
    # Class distribution
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

        print(
            f"{group:4} -> {count} samples"
        )

    # --------------------------------------------------
    # Cross-validation
    # --------------------------------------------------

    print("\n[3] CROSS-VALIDATION")

    min_class_count = int(
        class_counts.min()
    )

    n_splits = min(
        N_SPLITS,
        min_class_count,
    )

    if n_splits < 2:

        raise ValueError(
            "At least two samples per class are required "
            "for stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"Strategy        : StratifiedKFold"
    )

    print(
        f"Folds           : {n_splits}"
    )

    print(
        f"Random state     : {RANDOM_STATE}"
    )

    # --------------------------------------------------
    # Feature selection experiments
    # --------------------------------------------------

    feature_counts = [
        10,
        20,
        30,
        50,
    ]

    print("\n[4] FEATURE SELECTION")

    print(
        f"Top-k values    : {feature_counts}"
    )

    all_results = []

    # --------------------------------------------------
    # Experiments
    # --------------------------------------------------

    for k in feature_counts:

        print("\n" + "-" * 70)

        print(
            f"Top-{k} Features + SVM"
        )

        print("-" * 70)

        fold_results = []

        for fold_number, (
            train_idx,
            test_idx,
        ) in enumerate(
            cv.split(X_features, y),
            start=1,
        ):

            X_train = X_features.iloc[
                train_idx
            ]

            X_test = X_features.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            y_test = y.iloc[
                test_idx
            ]

            actual_k = min(
                k,
                X_train.shape[1],
            )

            # ------------------------------------------
            # IMPORTANT:
            #
            # SelectKBest is fitted ONLY on training
            # fold.
            #
            # Therefore the feature selection itself
            # cannot leak validation information.
            # ------------------------------------------

            pipeline = Pipeline(
                steps=[
                    (
                        "feature_selection",
                        SelectKBest(
                            score_func=mutual_info_classif,
                            k=actual_k,
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                        ),
                    ),
                ]
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

        # ----------------------------------------------
        # Summary
        # ----------------------------------------------

        fold_df = pd.DataFrame(
            fold_results
        )

        print("\nExperiment summary:")

        print(
            f"Accuracy          : "
            f"{fold_df['Accuracy'].mean():.4f} "
            f"± "
            f"{fold_df['Accuracy'].std(ddof=0):.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{fold_df['BalancedAccuracy'].mean():.4f} "
            f"± "
            f"{fold_df['BalancedAccuracy'].std(ddof=0):.4f}"
        )

        print(
            f"Macro F1          : "
            f"{fold_df['F1Macro'].mean():.4f} "
            f"± "
            f"{fold_df['F1Macro'].std(ddof=0):.4f}"
        )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 70)
    print("[5] FEATURE-SVM RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )

    print("\n" + "=" * 70)
    print("ENGINEERED FEATURES + SVM FINISHED")
    print("=" * 70)

    return results_df