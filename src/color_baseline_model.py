from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# ============================================================
# CONFIG
# ============================================================

X_PATH = Path(
    "data/processed/color_X.npy"
)

PPM_PATH = Path(
    "data/processed/color_ppm.npy"
)

META_PATH = Path(
    "data/processed/color_preprocessed.csv"
)

OUTPUT_DIR = Path(
    "reports/color_baseline_model"
)

COLOR_REGION = (
    5.0,
    9.0,
)


# ============================================================
# LOAD
# ============================================================

def load_data():
    X = np.load(X_PATH)
    ppm = np.load(PPM_PATH)
    meta = pd.read_csv(META_PATH)

    return X, ppm, meta


# ============================================================
# DUPLICATE GROUPS
# ============================================================

def build_duplicate_groups(X):
    """
    Exact identical spectra receive the same CV group.
    This prevents an identical spectrum from appearing in both
    train and test folds.
    """

    groups = np.empty(
        X.shape[0],
        dtype=int,
    )

    signatures = {}
    next_group = 0

    for i, spectrum in enumerate(X):

        signature = spectrum.tobytes()

        if signature not in signatures:
            signatures[signature] = next_group
            next_group += 1

        groups[i] = signatures[signature]

    return groups


# ============================================================
# MODELS
# ============================================================

def build_models():

    models = {

        "logistic_regression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "pca",
                    PCA(
                        n_components=0.95,
                    ),
                ),

                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        "linear_svm": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "pca",
                    PCA(
                        n_components=0.95,
                    ),
                ),

                (
                    "model",
                    SVC(
                        kernel="linear",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),

        "rbf_svm": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "pca",
                    PCA(
                        n_components=0.95,
                    ),
                ),

                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# CROSS VALIDATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    groups,
    sample_meta,
):

    logo = LeaveOneGroupOut()

    predictions = np.full(
        len(y),
        np.nan,
    )

    probabilities = np.full(
        len(y),
        np.nan,
    )

    fold_count = 0

    for train_idx, test_idx in logo.split(
        X,
        y,
        groups,
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]

        model.fit(
            X_train,
            y_train,
        )

        predictions[test_idx] = (
            model.predict(
                X_test
            )
        )

        if hasattr(
            model,
            "predict_proba",
        ):
            probabilities[
                test_idx
            ] = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

        fold_count += 1

    valid = (
        ~np.isnan(predictions)
    )

    y_true = y[
        valid
    ]

    y_pred = predictions[
        valid
    ].astype(int)

    y_prob = probabilities[
        valid
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        auc = np.nan

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "no_artificial_color",
            "artificial_color",
        ],
        zero_division=0,
    )

    prediction_rows = []

    for index in range(
        len(y)
    ):

        prediction_rows.append(
            {
                "number":
                    sample_meta.iloc[index][
                        "number"
                    ],

                "name":
                    sample_meta.iloc[index][
                        "name"
                    ],

                "true_artificial_color":
                    int(y[index]),

                "predicted_artificial_color":
                    int(y_pred[index]),

                "probability_artificial_color":
                    float(
                        y_prob[index]
                    ),

                "correct":
                    bool(
                        y[index]
                        == y_pred[index]
                    ),
            }
        )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    return {
        "folds": fold_count,
        "accuracy": float(accuracy),
        "balanced_accuracy":
            float(balanced_accuracy),
        "roc_auc":
            float(auc)
            if not np.isnan(auc)
            else None,
        "confusion_matrix":
            matrix.tolist(),
        "classification_report":
            report,
        "predictions":
            predictions_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR ARTIFICIAL-COLOR BASELINE MODEL")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X, ppm, meta = load_data()

    print()
    print(
        f"X shape : {X.shape}"
    )

    # --------------------------------------------------------
    # COLOR REGION
    # --------------------------------------------------------

    region_mask = (
        (ppm >= COLOR_REGION[0])
        & (ppm <= COLOR_REGION[1])
    )

    X_region = X[
        :,
        region_mask,
    ]

    print(
        f"5–9 ppm matrix : "
        f"{X_region.shape}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = (
        meta[
            "artificial_color"
        ]
        .astype(int)
        .to_numpy()
    )

    print()
    print("[1] TARGET")

    print(
        f"No artificial color : "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Artificial color    : "
        f"{np.sum(y == 1)}"
    )

    # --------------------------------------------------------
    # DUPLICATE-AWARE CV
    # --------------------------------------------------------

    groups = build_duplicate_groups(
        X_region
    )

    print()
    print("[2] CROSS-VALIDATION")

    print(
        f"Unique CV groups : "
        f"{len(np.unique(groups))}"
    )

    duplicate_count = (
        len(groups)
        - len(np.unique(groups))
    )

    print(
        f"Duplicate-linked samples : "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = build_models()

    all_results = []

    for model_name, model in models.items():

        print()
        print(
            f"[3] MODEL: {model_name}"
        )

        result = evaluate_model(
            model,
            X_region,
            y,
            groups,
            meta,
        )

        print(
            f"Accuracy          : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Balanced accuracy : "
            f"{result['balanced_accuracy']:.4f}"
        )

        if result["roc_auc"] is not None:

            print(
                f"ROC-AUC           : "
                f"{result['roc_auc']:.4f}"
            )

        print()
        print(
            "Confusion matrix:"
        )

        print(
            np.asarray(
                result["confusion_matrix"]
            )
        )

        print()
        print(
            result[
                "classification_report"
            ]
        )

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        prediction_path = (
            OUTPUT_DIR
            / f"{model_name}_predictions.csv"
        )

        result[
            "predictions"
        ].to_csv(
            prediction_path,
            index=False,
            encoding="utf-8-sig",
        )

        all_results.append(
            {
                "model":
                    model_name,

                "accuracy":
                    result["accuracy"],

                "balanced_accuracy":
                    result[
                        "balanced_accuracy"
                    ],

                "roc_auc":
                    result["roc_auc"],
            }
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame(
        all_results
    )

    summary = summary.sort_values(
        "balanced_accuracy",
        ascending=False,
    )

    summary_path = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {summary_path}"
    )

    print()
    print("=" * 70)
    print("BASELINE MODELING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()