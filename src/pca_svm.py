import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def run_pca_svm(
    X: pd.DataFrame,
    y: pd.Series,
    results_path: str = "pca_svm_results.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PCA + SVM LEAKAGE-SAFE MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Input information
    # --------------------------------------------------

    print("\n[1] INPUT DATA")

    print(f"Samples         : {X.shape[0]}")
    print(f"Spectral points : {X.shape[1]}")
    print(f"Classes         : {y.nunique()}")

    # --------------------------------------------------
    # 2. Cross-validation
    # --------------------------------------------------

    print("\n[2] CROSS-VALIDATION")

    cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    print("Strategy         : StratifiedKFold")
    print("Number of folds  : 2")
    print("Shuffle          : True")
    print("Random state     : 42")

    # --------------------------------------------------
    # 3. Leakage-safe pipeline
    # --------------------------------------------------
    #
    # IMPORTANT:
    #
    # StandardScaler is fitted only on training data
    # of each fold.
    #
    # PCA is also fitted only on training data
    # of each fold.
    #
    # SVM is trained only after the fold-specific
    # scaler and PCA are fitted.
    #
    # Validation data is NEVER used to fit scaler/PCA.
    #
    # --------------------------------------------------

    print("\n[3] MODEL PIPELINE")

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=0.95,
                    svd_solver="full",
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                ),
            ),
        ]
    )

    print("Scaler           : StandardScaler")
    print("PCA              : PCA")
    print("PCA variance     : 95%")
    print("PCA solver       : full")
    print("Classifier       : SVM")
    print("Kernel           : RBF")
    print("C                : 1.0")
    print("Gamma            : scale")

    # --------------------------------------------------
    # 4. Cross-validation loop
    # --------------------------------------------------

    print("\n[4] RUNNING CROSS-VALIDATION")

    fold_results = []

    X_values = X.to_numpy()
    y_values = y.to_numpy()

    for fold_number, (train_index, test_index) in enumerate(
        cv.split(X_values, y_values),
        start=1,
    ):

        X_train = X_values[train_index]
        X_test = X_values[test_index]

        y_train = y_values[train_index]
        y_test = y_values[test_index]

        # Fit pipeline ONLY on training fold.
        pipeline.fit(
            X_train,
            y_train,
        )

        # Predict untouched validation fold.
        y_pred = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred,
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # Number of PCA components selected in THIS fold.
        pca_components = pipeline.named_steps[
            "pca"
        ].n_components_

        fold_results.append(
            {
                "Fold": fold_number,
                "Accuracy": accuracy,
                "BalancedAccuracy": balanced_accuracy,
                "F1Macro": macro_f1,
                "PCAComponents": pca_components,
            }
        )

        print(
            f"Fold {fold_number}: "
            f"Accuracy={accuracy:.4f}, "
            f"BalancedAccuracy={balanced_accuracy:.4f}, "
            f"F1Macro={macro_f1:.4f}, "
            f"PCAComponents={pca_components}"
        )

    # --------------------------------------------------
    # 5. Results table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        fold_results
    )

    print("\n[5] FOLD RESULTS")

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    print("\n[6] SUMMARY")

    accuracy_mean = results_df[
        "Accuracy"
    ].mean()

    accuracy_std = results_df[
        "Accuracy"
    ].std()

    balanced_mean = results_df[
        "BalancedAccuracy"
    ].mean()

    balanced_std = results_df[
        "BalancedAccuracy"
    ].std()

    f1_mean = results_df[
        "F1Macro"
    ].mean()

    f1_std = results_df[
        "F1Macro"
    ].std()

    print(
        f"Accuracy          : "
        f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_mean:.4f} ± {balanced_std:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{f1_mean:.4f} ± {f1_std:.4f}"
    )

    print(
        f"Average PCA components : "
        f"{results_df['PCAComponents'].mean():.2f}"
    )

    # --------------------------------------------------
    # 7. Save results
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
    print("PCA + SVM EVALUATION FINISHED")
    print("=" * 70)

    return results_df