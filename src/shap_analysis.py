import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

import shap


RANDOM_STATE = 42

# Must match the current best robust candidate.
TOP_K = 50


def run_shap_analysis(
    X_features: pd.DataFrame,
    y: pd.Series,
    output_path: str = "shap_feature_importance.csv",
    plot_path: str = "shap_summary.png",
) -> pd.DataFrame:

    print(
        "\n" + "=" * 70
    )

    print(
        "SHAP FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------
    # 1. Validation
    # --------------------------------------------------

    if X_features.empty:
        raise ValueError(
            "Feature matrix is empty."
        )

    if len(X_features) != len(y):
        raise ValueError(
            "X_features and y row counts do not match."
        )

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    # --------------------------------------------------
    # 2. Encode target
    # --------------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y.astype(str)
    )

    class_count = len(
        encoder.classes_
    )

    if class_count < 2:
        raise ValueError(
            "At least two classes are required."
        )

    actual_top_k = min(
        TOP_K,
        X_features.shape[1],
    )

    print(
        "\n[1] INPUT DATA"
    )

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
        f"{class_count}"
    )

    print(
        f"Selected Top-K      : "
        f"{actual_top_k}"
    )

    # --------------------------------------------------
    # 3. Feature selection
    # --------------------------------------------------

    selector = SelectKBest(
        score_func=lambda X, target: (
            mutual_info_classif(
                X,
                target,
                random_state=RANDOM_STATE,
            )
        ),
        k=actual_top_k,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        X_selected = (
            selector.fit_transform(
                X_features,
                y_encoded,
            )
        )

    selected_mask = (
        selector.get_support()
    )

    selected_features = (
        X_features.columns[
            selected_mask
        ]
    )

    X_selected_df = (
        pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X_features.index,
        )
    )

    print(
        f"Selected features   : "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------
    # 4. Final reference model
    # --------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )

    model.fit(
        X_selected_df,
        y_encoded,
    )

    # --------------------------------------------------
    # 5. SHAP values
    # --------------------------------------------------

    print(
        "\n[2] CALCULATING SHAP VALUES"
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_selected_df
        )
    )

    # --------------------------------------------------
    # 6. Handle SHAP output formats
    # --------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):

        # Older SHAP multiclass format:
        # list[class] -> (samples, features)

        absolute_values = np.mean(
            [
                np.abs(values)
                for values in shap_values
            ],
            axis=0,
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            # Newer format:
            # (samples, features, classes)

            absolute_values = np.mean(
                np.abs(
                    shap_array
                ),
                axis=2,
            )

        elif shap_array.ndim == 2:

            absolute_values = np.abs(
                shap_array
            )

        else:

            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )

    mean_abs_shap = np.mean(
        absolute_values,
        axis=0,
    )

    # --------------------------------------------------
    # 7. Importance table
    # --------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "Feature":
                selected_features,
            "MeanAbsSHAP":
                mean_abs_shap,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="MeanAbsSHAP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(importance_df) + 1,
        ),
    )

    # --------------------------------------------------
    # 8. Save importance
    # --------------------------------------------------

    importance_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    # --------------------------------------------------
    # 9. SHAP summary plot
    # --------------------------------------------------

    print(
        "\n[3] GENERATING SHAP PLOT"
    )

    top_plot = (
        importance_df.head(20)
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        top_plot[
            "Feature"
        ][::-1],
        top_plot[
            "MeanAbsSHAP"
        ][::-1],
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Feature Importance - "
        "XGBoost Top-50 Candidate"
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Plot saved to: "
        f"{plot_path}"
    )

    # --------------------------------------------------
    # 10. Top features
    # --------------------------------------------------

    print(
        "\n[4] TOP FEATURES"
    )

    print(
        importance_df.head(20).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SHAP ANALYSIS FINISHED"
    )

    print(
        "=" * 70
    )

    return importance_df