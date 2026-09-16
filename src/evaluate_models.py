import pandas as pd

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


def evaluate_svm(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("SVM BASELINE MODEL EVALUATION")
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
    # 3. Cross Validation
    # --------------------------------------------------

    print("\n[3] CROSS-VALIDATION")

    min_class_count = int(class_counts.min())

    n_splits = min(N_SPLITS, min_class_count)

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

    print(f"Strategy         : StratifiedKFold")
    print(f"Number of folds  : {n_splits}")
    print(f"Shuffle          : True")
    print(f"Random state     : {RANDOM_STATE}")

    # --------------------------------------------------
    # 4. Model pipeline
    # --------------------------------------------------
    #
    # IMPORTANT:
    # StandardScaler is inside the Pipeline.
    #
    # Therefore the scaler is fitted separately
    # inside each training fold.
    #
    # This prevents data leakage.
    # --------------------------------------------------

    print("\n[4] MODEL PIPELINE")

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                )
            ),
        ]
    )

    print("Scaler           : StandardScaler")
    print("Classifier       : SVM")
    print("Kernel           : RBF")
    print("C                : 1.0")
    print("Gamma            : scale")

    # --------------------------------------------------
    # 5. Cross-validation
    # --------------------------------------------------

    print("\n[5] RUNNING CROSS-VALIDATION")

    results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append(
            {
                "Fold": fold_number,
                "Accuracy": accuracy,
                "BalancedAccuracy": balanced_accuracy,
                "F1Macro": macro_f1,
            }
        )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # 6. Fold results
    # --------------------------------------------------

    print("\nFold results:")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 7. Summary
    # --------------------------------------------------

    summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
            ],
            "Mean": [
                results_df["Accuracy"].mean(),
                results_df["BalancedAccuracy"].mean(),
                results_df["F1Macro"].mean(),
            ],
            "Std": [
                results_df["Accuracy"].std(
                    ddof=0
                ),
                results_df["BalancedAccuracy"].std(
                    ddof=0
                ),
                results_df["F1Macro"].std(
                    ddof=0
                ),
            ],
        }
    )

    print("\n[6] SUMMARY")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 8. Save results
    # --------------------------------------------------

    output_path = "svm_baseline_results.csv"

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nResults saved to: {output_path}"
    )

    print("\n" + "=" * 70)
    print("SVM BASELINE EVALUATION FINISHED")
    print("=" * 70)

    return summary