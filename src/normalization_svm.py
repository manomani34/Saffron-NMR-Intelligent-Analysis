import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
N_SPLITS = 2


def evaluate_normalization_svm(
    X: pd.DataFrame,
    y: pd.Series,
    results_path: str = "normalization_svm_results.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("NMR NORMALIZATION + SVM EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Input information
    # --------------------------------------------------

    print("\n[1] INPUT DATA")

    print(f"Samples         : {X.shape[0]}")
    print(f"Spectral points : {X.shape[1]}")
    print(f"Classes         : {y.nunique()}")

    # --------------------------------------------------
    # 2. Class distribution
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
    # 3. Cross-validation
    # --------------------------------------------------

    print("\n[3] CROSS-VALIDATION")

    min_class_count = int(class_counts.min())

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
        f"Strategy         : StratifiedKFold"
    )

    print(
        f"Number of folds  : {n_splits}"
    )

    print(
        f"Shuffle          : True"
    )

    print(
        f"Random state     : {RANDOM_STATE}"
    )

    # --------------------------------------------------
    # 4. Define experiments
    # --------------------------------------------------
    #
    # Experiment 1:
    # No sample normalization
    #
    # Experiment 2:
    # L1 normalization
    #
    # Experiment 3:
    # L2 normalization
    #
    # StandardScaler remains after sample normalization.
    #
    # This allows us to separately control:
    #
    #   sample-level scaling
    #   feature-level scaling
    #
    # --------------------------------------------------

    experiments = {
        "Raw": None,

        "L1": Normalizer(
            norm="l1"
        ),

        "L2": Normalizer(
            norm="l2"
        ),
    }

    # --------------------------------------------------
    # 5. Run experiments
    # --------------------------------------------------

    print("\n[4] RUNNING NORMALIZATION EXPERIMENTS")

    all_results = []

    for experiment_name, normalizer in experiments.items():

        print("\n" + "-" * 70)

        print(
            f"Experiment: {experiment_name}"
        )

        print("-" * 70)

        steps = []

        # Sample-level normalization
        if normalizer is not None:

            steps.append(
                (
                    "normalizer",
                    normalizer,
                )
            )

        # Feature-level standardization
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

        # SVM classifier
        steps.append(
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                )
            )
        )

        pipeline = Pipeline(
            steps=steps
        )

        if normalizer is None:

            print(
                "Sample normalization : None"
            )

        else:

            print(
                f"Sample normalization : "
                f"{experiment_name}"
            )

        print(
            "Feature scaling      : StandardScaler"
        )

        print(
            "Classifier            : SVM-RBF"
        )

        fold_results = []

        for fold_number, (
            train_idx,
            test_idx,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # ------------------------------------------
            # Fit only on training fold
            # ------------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------
            # Predict untouched validation fold
            # ------------------------------------------

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

            fold_result = {
                "Experiment": experiment_name,
                "Fold": fold_number,
                "Accuracy": accuracy,
                "BalancedAccuracy": balanced_accuracy,
                "F1Macro": macro_f1,
            }

            fold_results.append(
                fold_result
            )

            all_results.append(
                fold_result
            )

            print(
                f"Fold {fold_number}: "
                f"Accuracy={accuracy:.4f}, "
                f"BalancedAccuracy={balanced_accuracy:.4f}, "
                f"F1Macro={macro_f1:.4f}"
            )

        # ----------------------------------------------
        # Experiment summary
        # ----------------------------------------------

        experiment_df = pd.DataFrame(
            fold_results
        )

        print("\nExperiment summary:")

        print(
            f"Accuracy          : "
            f"{experiment_df['Accuracy'].mean():.4f} "
            f"± "
            f"{experiment_df['Accuracy'].std(ddof=0):.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{experiment_df['BalancedAccuracy'].mean():.4f} "
            f"± "
            f"{experiment_df['BalancedAccuracy'].std(ddof=0):.4f}"
        )

        print(
            f"Macro F1          : "
            f"{experiment_df['F1Macro'].mean():.4f} "
            f"± "
            f"{experiment_df['F1Macro'].std(ddof=0):.4f}"
        )

    # --------------------------------------------------
    # 6. Create final results table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 70)
    print("[5] ALL EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 7. Aggregate results
    # --------------------------------------------------

    summary_df = (
        results_df
        .groupby("Experiment")
        .agg(
            AccuracyMean=("Accuracy", "mean"),
            AccuracyStd=("Accuracy", "std"),
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

    # With only two folds, pandas std uses sample std.
    # Fill possible NaN values defensively.
    summary_df = summary_df.fillna(0.0)

    # --------------------------------------------------
    # 8. Print summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] NORMALIZATION COMPARISON")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 9. Save results
    # --------------------------------------------------

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nDetailed results saved to: "
        f"{results_path}"
    )

    summary_path = (
        results_path
        .replace(
            ".csv",
            "_summary.csv",
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("\n" + "=" * 70)
    print("NORMALIZATION + SVM EXPERIMENT FINISHED")
    print("=" * 70)

    return summary_df