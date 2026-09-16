import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 2


def evaluate_pls_da(
    X: pd.DataFrame,
    y: pd.Series,
    results_path: str = "pls_da_results.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PLS-DA BASELINE MODEL EVALUATION")
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
        "Strategy         : StratifiedKFold"
    )

    print(
        f"Number of folds  : {n_splits}"
    )

    print(
        "Shuffle           : True"
    )

    print(
        f"Random state      : {RANDOM_STATE}"
    )

    # --------------------------------------------------
    # 4. Class encoding
    # --------------------------------------------------

    print("\n[4] CLASS ENCODING")

    classes = np.array(
        sorted(
            y.unique(),
            key=lambda value:
                int(str(value)[1:])
                if str(value).startswith("G")
                else 999,
        )
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    print(
        f"Number of classes : {len(classes)}"
    )

    print(
        f"Classes           : {classes.tolist()}"
    )

    # --------------------------------------------------
    # 5. Experiments
    # --------------------------------------------------
    #
    # We do NOT choose the number of components
    # according to test performance.
    #
    # Instead, several fixed component counts are
    # evaluated as separate exploratory experiments.
    #
    # --------------------------------------------------

    component_values = [
        2,
        3,
        5,
    ]

    print(
        "\n[5] PLS COMPONENT EXPERIMENTS"
    )

    print(
        f"Components tested : {component_values}"
    )

    all_results = []

    # --------------------------------------------------
    # 6. Cross-validation
    # --------------------------------------------------

    for n_components in component_values:

        print("\n" + "-" * 70)

        print(
            f"PLS-DA Components = {n_components}"
        )

        print("-" * 70)

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
            # Convert classes to integer indices
            # ------------------------------------------

            y_train_indices = np.array([
                class_to_index[value]
                for value in y_train
            ])

            y_test_indices = np.array([
                class_to_index[value]
                for value in y_test
            ])

            # ------------------------------------------
            # One-hot encoding
            # ------------------------------------------

            y_train_onehot = np.zeros(
                (
                    len(y_train_indices),
                    len(classes),
                )
            )

            y_train_onehot[
                np.arange(len(y_train_indices)),
                y_train_indices,
            ] = 1.0

            # ------------------------------------------
            # PLS-DA pipeline
            # ------------------------------------------

            pipeline = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "pls",
                        PLSRegression(
                            n_components=n_components,
                            scale=False,
                            max_iter=500,
                        ),
                    ),
                ]
            )

            # ------------------------------------------
            # Fit on training fold only
            # ------------------------------------------

            pipeline.fit(
                X_train,
                y_train_onehot,
            )

            # ------------------------------------------
            # Predict validation fold
            # ------------------------------------------

            y_pred_scores = pipeline.predict(
                X_test
            )

            # ------------------------------------------
            # Convert regression outputs into class
            # prediction using argmax.
            # ------------------------------------------

            y_pred_indices = np.argmax(
                y_pred_scores,
                axis=1,
            )

            y_pred = classes[
                y_pred_indices
            ]

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
                "Components": n_components,
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
    # 7. Results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    print("\n" + "=" * 70)
    print("[6] PLS-DA RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 8. Summary by number of components
    # --------------------------------------------------

    summary_df = (
        results_df
        .groupby("Components")
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

    # --------------------------------------------------
    # 9. Print summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("[7] PLS-DA SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = (
        results_path.replace(
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
        f"\nDetailed results saved to: "
        f"{results_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("\n" + "=" * 70)
    print("PLS-DA EVALUATION FINISHED")
    print("=" * 70)

    return summary_df