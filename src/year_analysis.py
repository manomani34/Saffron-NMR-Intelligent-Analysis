# src/year_analysis.py

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# GENERAL HELPERS
# ============================================================

def _safe_metric_summary(y_true, y_pred):
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


def _build_svm_pipeline(k):
    return Pipeline(
        [
            (
                "feature_selection",
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=k,
                ),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=False,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_xgb_classifier(num_classes):
    if not XGBOOST_AVAILABLE:
        return None

    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


# ============================================================
# YEAR PCA
# ============================================================

def _run_year_pca(
    X,
    metadata,
    output_dir,
):
    print(
        "\n[5A] PCA ANALYSIS BY HARVEST YEAR"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    X_values = X.to_numpy(
        dtype=float
    )

    n_components = min(
        8,
        X_values.shape[0] - 1,
        X_values.shape[1],
    )

    pca = PCA(
        n_components=n_components,
        random_state=42,
    )

    scores = pca.fit_transform(
        X_values
    )

    score_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_components + 1,
        )
    ]

    score_df = pd.DataFrame(
        scores,
        columns=score_columns,
        index=metadata.index,
    )

    score_df["SampleId"] = (
        metadata["SampleId"].values
    )

    score_df["OriginalSampleName"] = (
        metadata["OriginalSampleName"].values
    )

    score_df["Group"] = (
        metadata["Group"]
        .astype(str)
        .values
    )

    score_df["HarvestYear"] = (
        metadata["HarvestYear"]
        .astype(int)
        .values
    )

    score_path = os.path.join(
        output_dir,
        "year_pca_scores.csv",
    )

    score_df.to_csv(
        score_path,
        index=False,
        encoding="utf-8-sig",
    )

    explained = (
        pca.explained_variance_ratio_
    )

    explained_df = pd.DataFrame(
        {
            "PC": score_columns,
            "ExplainedVarianceRatio": explained,
            "ExplainedVariancePercent":
                explained * 100.0,
            "CumulativeVariancePercent":
                np.cumsum(explained) * 100.0,
        }
    )

    explained_path = os.path.join(
        output_dir,
        "year_pca_explained_variance.csv",
    )

    explained_df.to_csv(
        explained_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nExplained variance:"
    )

    print(
        explained_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PCA by year
    # --------------------------------------------------------

    if n_components >= 2:

        plt.figure(
            figsize=(10, 7)
        )

        years = sorted(
            score_df[
                "HarvestYear"
            ].unique()
        )

        markers = {
            1394: "o",
            1404: "^",
        }

        for year in years:

            mask = (
                score_df[
                    "HarvestYear"
                ]
                == year
            )

            plt.scatter(
                score_df.loc[
                    mask,
                    "PC1",
                ],
                score_df.loc[
                    mask,
                    "PC2",
                ],
                marker=markers.get(
                    int(year),
                    "o",
                ),
                s=80,
                alpha=0.8,
                label=str(
                    int(year)
                ),
            )

        plt.xlabel(
            f"PC1 "
            f"({explained[0] * 100:.2f}%)"
        )

        plt.ylabel(
            f"PC2 "
            f"({explained[1] * 100:.2f}%)"
        )

        plt.title(
            "PCA by Harvest Year"
        )

        plt.legend(
            title="Harvest Year"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plot_path = os.path.join(
            output_dir,
            "year_pca_plot.png",
        )

        plt.savefig(
            plot_path,
            dpi=200,
        )

        plt.close()

        print(
            f"\nSaved PCA plot: "
            f"{plot_path}"
        )

    # --------------------------------------------------------
    # Combined region + year PCA
    # --------------------------------------------------------

    if n_components >= 2:

        plt.figure(
            figsize=(12, 8)
        )

        groups = sorted(
            score_df[
                "Group"
            ].unique()
        )

        color_map = (
            plt.colormaps["tab20"]
            .resampled(
                max(
                    len(groups),
                    1,
                )
            )
        )

        group_to_color = {
            group: color_map(i)
            for i, group in enumerate(
                groups
            )
        }

        for _, row in (
            score_df.iterrows()
        ):

            marker = markers.get(
                int(
                    row[
                        "HarvestYear"
                    ]
                ),
                "o",
            )

            plt.scatter(
                row["PC1"],
                row["PC2"],
                marker=marker,
                s=85,
                alpha=0.85,
                color=group_to_color[
                    row["Group"]
                ],
            )

            plt.text(
                row["PC1"],
                row["PC2"],
                str(
                    row["Group"]
                ),
                fontsize=7,
                alpha=0.8,
            )

        plt.xlabel(
            f"PC1 "
            f"({explained[0] * 100:.2f}%)"
        )

        plt.ylabel(
            f"PC2 "
            f"({explained[1] * 100:.2f}%)"
        )

        plt.title(
            "PCA: Region + Harvest Year"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        combined_plot_path = os.path.join(
            output_dir,
            "year_region_pca_plot.png",
        )

        plt.savefig(
            combined_plot_path,
            dpi=200,
        )

        plt.close()

        print(
            f"Saved combined PCA plot: "
            f"{combined_plot_path}"
        )

    return score_df, explained_df


# ============================================================
# YEAR PREDICTION WITH CROSS-VALIDATION
# ============================================================

def _run_year_prediction_cv(
    X,
    metadata,
    output_dir,
):
    print(
        "\n[5B] HARVEST YEAR PREDICTION "
        "WITH CROSS-VALIDATION"
    )

    years = (
        metadata["HarvestYear"]
        .astype(int)
        .astype(str)
    )

    encoder = LabelEncoder()

    y_year = encoder.fit_transform(
        years
    )

    n_classes = len(
        encoder.classes_
    )

    if n_classes < 2:
        raise ValueError(
            "At least two harvest years "
            "are required."
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    k = min(
        30,
        X.shape[1],
    )

    # --------------------------------------------------------
    # Repeated Stratified CV
    # --------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42,
    )

    fold_results = []

    print(
        "\nRunning 5-fold CV × 10 repeats..."
    )

    for fold_index, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y_year,
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y_year[
            train_idx
        ]

        y_test = y_year[
            test_idx
        ]

        model = Pipeline(
            [
                (
                    "feature_selection",
                    SelectKBest(
                        score_func=
                            mutual_info_classif,
                        k=k,
                    ),
                ),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        metrics = _safe_metric_summary(
            y_test,
            y_pred,
        )

        fold_results.append(
            {
                "Fold": fold_index,
                "TrainSamples":
                    len(train_idx),
                "TestSamples":
                    len(test_idx),
                "Accuracy":
                    metrics["Accuracy"],
                "BalancedAccuracy":
                    metrics[
                        "BalancedAccuracy"
                    ],
                "F1Macro":
                    metrics[
                        "F1Macro"
                    ],
            }
        )

    fold_df = pd.DataFrame(
        fold_results
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    mean_accuracy = (
        fold_df[
            "Accuracy"
        ].mean()
    )

    std_accuracy = (
        fold_df[
            "Accuracy"
        ].std(
            ddof=1
        )
    )

    mean_ba = (
        fold_df[
            "BalancedAccuracy"
        ].mean()
    )

    std_ba = (
        fold_df[
            "BalancedAccuracy"
        ].std(
            ddof=1
        )
    )

    mean_f1 = (
        fold_df[
            "F1Macro"
        ].mean()
    )

    std_f1 = (
        fold_df[
            "F1Macro"
        ].std(
            ddof=1
        )
    )

    chance_accuracy = (
        1.0
        / n_classes
    )

    chance_balanced_accuracy = (
        1.0
        / n_classes
    )

    print(
        "\nHarvest-year prediction "
        "cross-validation results:"
    )

    print(
        f"Accuracy             : "
        f"{mean_accuracy:.4f} "
        f"± {std_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy    : "
        f"{mean_ba:.4f} "
        f"± {std_ba:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{mean_f1:.4f} "
        f"± {std_f1:.4f}"
    )

    print(
        f"Chance Accuracy      : "
        f"{chance_accuracy:.4f}"
    )

    print(
        f"Chance Balanced Acc. : "
        f"{chance_balanced_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save fold results
    # --------------------------------------------------------

    fold_results_path = os.path.join(
        output_dir,
        "harvest_year_prediction_cv_folds.csv",
    )

    fold_df.to_csv(
        fold_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "AccuracyMean":
            mean_accuracy,
        "AccuracyStd":
            std_accuracy,
        "BalancedAccuracyMean":
            mean_ba,
        "BalancedAccuracyStd":
            std_ba,
        "F1MacroMean":
            mean_f1,
        "F1MacroStd":
            std_f1,
        "ChanceAccuracy":
            chance_accuracy,
        "ChanceBalancedAccuracy":
            chance_balanced_accuracy,
        "Samples":
            len(y_year),
        "CVFolds":
            5,
        "CVRepeats":
            10,
        "TotalEvaluations":
            len(fold_df),
        "TopK":
            k,
    }

    summary_path = os.path.join(
        output_dir,
        "harvest_year_prediction_cv.csv",
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved: "
        f"{fold_results_path}"
    )

    print(
        f"Saved: "
        f"{summary_path}"
    )

    # --------------------------------------------------------
    # Scientific interpretation
    # --------------------------------------------------------

    # For binary year classification,
    # balanced accuracy chance level is 0.5.
    binary_chance_ba = 0.5

    if mean_ba > binary_chance_ba:
        print(
            "\nINTERPRETATION:"
            "\nCross-validated spectral "
            "prediction of harvest year "
            "is above the binary chance level."
        )

    else:
        print(
            "\nINTERPRETATION:"
            "\nCross-validated harvest-year "
            "prediction is not convincingly "
            "above the binary chance level."
        )

    return {
        "Accuracy":
            mean_accuracy,
        "BalancedAccuracy":
            mean_ba,
        "F1Macro":
            mean_f1,
        "ChanceAccuracy":
            chance_accuracy,
        "ChanceBalancedAccuracy":
            chance_balanced_accuracy,
        "Samples":
            len(y_year),
        "CVFolds":
            5,
        "CVRepeats":
            10,
        "TopK":
            k,
    }


# ============================================================
# WITHIN-REGION YEAR EFFECT
# ============================================================

def _within_region_statistic(
    scores,
    regions,
    years,
):
    """
    Measures between-year separation
    within regions using Euclidean
    distance between year centroids
    in PCA space.

    This is a descriptive statistic used
    with a permutation test.
    """

    scores = np.asarray(
        scores,
        dtype=float,
    )

    regions = np.asarray(
        regions,
    )

    years = np.asarray(
        years,
    )

    total_stat = 0.0

    used_regions = 0

    for region in sorted(
        np.unique(
            regions
        )
    ):

        region_mask = (
            regions
            == region
        )

        region_scores = (
            scores[
                region_mask
            ]
        )

        region_years = (
            years[
                region_mask
            ]
        )

        unique_years = np.unique(
            region_years
        )

        if len(
            unique_years
        ) != 2:
            continue

        year_values = []

        valid = True

        for year in unique_years:

            mask = (
                region_years
                == year
            )

            if mask.sum() == 0:
                valid = False
                break

            year_values.append(
                region_scores[
                    mask
                ].mean(
                    axis=0
                )
            )

        if not valid:
            continue

        difference = (
            year_values[0]
            - year_values[1]
        )

        total_stat += float(
            np.sum(
                difference ** 2
            )
        )

        used_regions += 1

    if used_regions == 0:
        return np.nan

    return total_stat


def _run_within_region_year_permutation(
    pca_scores,
    metadata,
    output_dir,
    n_permutations=2000,
):
    print(
        "\n[5C] WITHIN-REGION HARVEST YEAR EFFECT"
    )

    common_regions = []

    region_values = (
        metadata["Group"]
        .astype(str)
        .values
    )

    year_values = (
        metadata["HarvestYear"]
        .astype(int)
        .values
    )

    for region in sorted(
        np.unique(
            region_values
        )
    ):

        region_years = np.unique(
            year_values[
                region_values
                == region
            ]
        )

        if set(
            region_years
        ) == {
            1394,
            1404,
        }:
            common_regions.append(
                region
            )

    print(
        "Common regions:"
    )

    print(
        ", ".join(
            common_regions
        )
    )

    if not common_regions:

        print(
            "No regions contain "
            "both years."
        )

        return {
            "ObservedStatistic":
                np.nan,
            "PermutationPValue":
                np.nan,
            "CommonRegions":
                0,
            "SamplesIncluded":
                0,
            "Permutations":
                0,
            "Interpretation":
                "Not enough common-region "
                "data for the test.",
        }

    mask = np.isin(
        region_values,
        common_regions,
    )

    scores = np.asarray(
        pca_scores.loc[
            mask
        ],
        dtype=float,
    )

    regions = region_values[
        mask
    ]

    years = year_values[
        mask
    ]

    observed = (
        _within_region_statistic(
            scores,
            regions,
            years,
        )
    )

    rng = np.random.default_rng(
        42
    )

    null_statistics = np.zeros(
        n_permutations,
        dtype=float,
    )

    for i in range(
        n_permutations
    ):

        shuffled_years = (
            years.copy()
        )

        # Shuffle year labels only
        # within each region.
        for region in common_regions:

            region_mask = (
                regions
                == region
            )

            region_years = (
                shuffled_years[
                    region_mask
                ].copy()
            )

            rng.shuffle(
                region_years
            )

            shuffled_years[
                region_mask
            ] = region_years

        null_statistics[i] = (
            _within_region_statistic(
                scores,
                regions,
                shuffled_years,
            )
        )

    p_value = (
        1.0
        + np.sum(
            null_statistics
            >= observed
        )
    ) / (
        n_permutations
        + 1.0
    )

    print(
        f"\nObserved statistic : "
        f"{observed:.6f}"
    )

    print(
        f"Permutation p-value: "
        f"{p_value:.4f}"
    )

    print(
        f"Permutations       : "
        f"{n_permutations}"
    )

    if p_value < 0.05:

        interpretation = (
            "Evidence of a within-region "
            "harvest-year spectral effect."
        )

    else:

        interpretation = (
            "No statistically convincing "
            "within-region harvest-year "
            "effect at alpha=0.05."
        )

    print(
        f"\nInterpretation: "
        f"{interpretation}"
    )

    result = {
        "ObservedStatistic":
            observed,
        "PermutationPValue":
            p_value,
        "CommonRegions":
            len(common_regions),
        "SamplesIncluded":
            len(scores),
        "Permutations":
            n_permutations,
        "Interpretation":
            interpretation,
    }

    summary_path = os.path.join(
        output_dir,
        "within_region_year_permutation.csv",
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Per-region descriptive effect
    # --------------------------------------------------------

    per_region_rows = []

    for region in common_regions:

        region_mask = (
            regions
            == region
        )

        region_scores = (
            scores[
                region_mask
            ]
        )

        region_years = (
            years[
                region_mask
            ]
        )

        year_1394_scores = (
            region_scores[
                region_years
                == 1394
            ]
        )

        year_1404_scores = (
            region_scores[
                region_years
                == 1404
            ]
        )

        mean_1394 = (
            year_1394_scores
            .mean(
                axis=0
            )
        )

        mean_1404 = (
            year_1404_scores
            .mean(
                axis=0
            )
        )

        distance = np.linalg.norm(
            mean_1394
            - mean_1404
        )

        per_region_rows.append(
            {
                "Group":
                    region,
                "N_1394":
                    int(
                        (
                            region_years
                            == 1394
                        ).sum()
                    ),
                "N_1404":
                    int(
                        (
                            region_years
                            == 1404
                        ).sum()
                    ),
                "PC_CentroidDistance":
                    distance,
            }
        )

    per_region_df = pd.DataFrame(
        per_region_rows
    )

    per_region_path = os.path.join(
        output_dir,
        "within_region_year_effect.csv",
    )

    per_region_df.to_csv(
        per_region_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {per_region_path}"
    )

    return result


# ============================================================
# CROSS-YEAR ORIGIN VALIDATION
# ============================================================

def _run_cross_year_direction(
    X,
    metadata,
    source_year,
    target_year,
    common_groups,
    output_dir,
):
    print(
        "\n"
        + "-" * 70
    )

    print(
        f"CROSS-YEAR ORIGIN VALIDATION: "
        f"{source_year} -> {target_year}"
    )

    print(
        "-" * 70
    )

    source_mask = (
        (
            metadata[
                "HarvestYear"
            ]
            == source_year
        )
        & metadata[
            "Group"
        ].isin(
            common_groups
        )
    )

    target_mask = (
        (
            metadata[
                "HarvestYear"
            ]
            == target_year
        )
        & metadata[
            "Group"
        ].isin(
            common_groups
        )
    )

    X_train = X.loc[
        source_mask
    ].copy()

    X_test = X.loc[
        target_mask
    ].copy()

    y_train = metadata.loc[
        source_mask,
        "Group",
    ].astype(str)

    y_test = metadata.loc[
        target_mask,
        "Group",
    ].astype(str)

    print(
        f"Training samples : "
        f"{len(X_train)}"
    )

    print(
        f"Test samples     : "
        f"{len(X_test)}"
    )

    train_distribution = (
        y_train
        .value_counts()
        .sort_index()
    )

    test_distribution = (
        y_test
        .value_counts()
        .sort_index()
    )

    print(
        "\nTraining class distribution:"
    )

    for group, count in (
        train_distribution.items()
    ):

        print(
            f"{group:<5} -> {count}"
        )

    print(
        "\nTest class distribution:"
    )

    for group, count in (
        test_distribution.items()
    ):

        print(
            f"{group:<5} -> {count}"
        )

    low_sample_classes = (
        train_distribution[
            train_distribution < 2
        ]
        .index
        .tolist()
    )

    if low_sample_classes:

        print(
            "\nWARNING:"
            "\nThe training year contains only one "
            "sample for these classes:"
        )

        print(
            "  "
            + ", ".join(
                low_sample_classes
            )
        )

        print(
            "\nCross-year evaluation is still performed, "
            "but these classes have extremely limited "
            "training representation."
        )

    label_encoder = LabelEncoder()

    y_train_encoded = (
        label_encoder.fit_transform(
            y_train
        )
    )

    unknown_test_classes = sorted(
        set(y_test)
        - set(
            label_encoder.classes_
        )
    )

    if unknown_test_classes:

        print(
            "\nERROR: Test contains classes "
            "that are not represented in training:"
        )

        print(
            "  "
            + ", ".join(
                unknown_test_classes
            )
        )

        return []

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    num_classes = len(
        label_encoder.classes_
    )

    print(
        f"\nTrain classes: "
        f"{num_classes}"
    )

    feature_counts = [
        10,
        20,
        30,
        50,
    ]

    results = []

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    print(
        "\nMODEL: SVM"
    )

    for k in feature_counts:

        actual_k = min(
            k,
            X_train.shape[1],
        )

        model = _build_svm_pipeline(
            actual_k
        )

        model.fit(
            X_train,
            y_train_encoded,
        )

        y_pred = model.predict(
            X_test
        )

        metrics = _safe_metric_summary(
            y_test_encoded,
            y_pred,
        )

        print(
            f"\nSVM + Top-{actual_k}"
        )

        print(
            f"Accuracy          : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{metrics['BalancedAccuracy']:.4f}"
        )

        print(
            f"Macro F1          : "
            f"{metrics['F1Macro']:.4f}"
        )

        results.append(
            {
                "SourceYear":
                    source_year,
                "TargetYear":
                    target_year,
                "Model":
                    "SVM",
                "TopK":
                    actual_k,
                "TrainSamples":
                    len(X_train),
                "TestSamples":
                    len(X_test),
                "Accuracy":
                    metrics[
                        "Accuracy"
                    ],
                "BalancedAccuracy":
                    metrics[
                        "BalancedAccuracy"
                    ],
                "F1Macro":
                    metrics[
                        "F1Macro"
                    ],
                "LowTrainingSampleClasses":
                    (
                        ",".join(
                            low_sample_classes
                        )
                        if low_sample_classes
                        else ""
                    ),
            }
        )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if XGBOOST_AVAILABLE:

        print(
            "\nMODEL: XGBoost"
        )

        for k in feature_counts:

            actual_k = min(
                k,
                X_train.shape[1],
            )

            pipeline = Pipeline(
                [
                    (
                        "feature_selection",
                        SelectKBest(
                            score_func=
                                mutual_info_classif,
                            k=actual_k,
                        ),
                    ),
                    (
                        "classifier",
                        _build_xgb_classifier(
                            num_classes
                        ),
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train_encoded,
            )

            y_pred = (
                pipeline.predict(
                    X_test
                )
            )

            metrics = (
                _safe_metric_summary(
                    y_test_encoded,
                    y_pred,
                )
            )

            print(
                f"\nXGBoost + Top-{actual_k}"
            )

            print(
                f"Accuracy          : "
                f"{metrics['Accuracy']:.4f}"
            )

            print(
                f"Balanced Accuracy : "
                f"{metrics['BalancedAccuracy']:.4f}"
            )

            print(
                f"Macro F1          : "
                f"{metrics['F1Macro']:.4f}"
            )

            results.append(
                {
                    "SourceYear":
                        source_year,
                    "TargetYear":
                        target_year,
                    "Model":
                        "XGBoost",
                    "TopK":
                        actual_k,
                    "TrainSamples":
                        len(X_train),
                    "TestSamples":
                        len(X_test),
                    "Accuracy":
                        metrics[
                            "Accuracy"
                        ],
                    "BalancedAccuracy":
                        metrics[
                            "BalancedAccuracy"
                        ],
                    "F1Macro":
                        metrics[
                            "F1Macro"
                        ],
                    "LowTrainingSampleClasses":
                        (
                            ",".join(
                                low_sample_classes
                            )
                            if low_sample_classes
                            else ""
                        ),
                }
            )

    else:

        print(
            "\nXGBoost is not available. "
            "Only SVM cross-year results "
            "will be produced."
        )

    return results


# ============================================================
# MAIN HARVEST YEAR ANALYSIS
# ============================================================

def analyze_harvest_year_effect(
    X,
    y,
    metadata,
    output_dir="outputs",
):
    print(
        "\n" + "=" * 70
    )

    print(
        "HARVEST YEAR EFFECT ANALYSIS"
    )

    print(
        "=" * 70
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    required_columns = {
        "HarvestYear",
        "OriginalSampleName",
        "SampleId",
    }

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:

        raise ValueError(
            "Missing required metadata columns: "
            f"{sorted(missing)}"
        )

    analysis_df = metadata.copy()

    analysis_df["Group"] = (
        y.astype(str).values
    )

    analysis_df["HarvestYear"] = (
        pd.to_numeric(
            analysis_df["HarvestYear"],
            errors="coerce",
        )
    )

    if (
        analysis_df["HarvestYear"]
        .isna()
        .any()
    ):

        raise ValueError(
            "Invalid HarvestYear values found."
        )

    # ========================================================
    # 1. YEAR DISTRIBUTION
    # ========================================================

    print(
        "\n[1] HARVEST YEAR DISTRIBUTION"
    )

    year_counts = (
        analysis_df[
            "HarvestYear"
        ]
        .value_counts()
        .sort_index()
    )

    for year, count in (
        year_counts.items()
    ):

        print(
            f"{int(year)} : {count}"
        )

    year_counts.to_csv(
        os.path.join(
            output_dir,
            "harvest_year_counts.csv",
        ),
        header=["Count"],
    )

    # ========================================================
    # 2. REGION × YEAR
    # ========================================================

    print(
        "\n[2] REGION × HARVEST YEAR"
    )

    region_year = pd.crosstab(
        analysis_df["Group"],
        analysis_df["HarvestYear"],
    )

    print(
        region_year.to_string()
    )

    region_year.to_csv(
        os.path.join(
            output_dir,
            "region_by_harvest_year.csv",
        )
    )

    # ========================================================
    # 3. COVERAGE
    # ========================================================

    print(
        "\n[3] YEAR COVERAGE PER REGION"
    )

    coverage_rows = []

    for group in sorted(
        analysis_df[
            "Group"
        ].unique()
    ):

        group_df = analysis_df[
            analysis_df["Group"]
            == group
        ]

        n_1394 = int(
            (
                group_df[
                    "HarvestYear"
                ]
                == 1394
            ).sum()
        )

        n_1404 = int(
            (
                group_df[
                    "HarvestYear"
                ]
                == 1404
            ).sum()
        )

        coverage_rows.append(
            {
                "Group":
                    group,
                "N":
                    len(group_df),
                "N_1394":
                    n_1394,
                "N_1404":
                    n_1404,
                "Has_1394":
                    n_1394 > 0,
                "Has_1404":
                    n_1404 > 0,
                "Both_Years":
                    (
                        n_1394 > 0
                        and n_1404 > 0
                    ),
            }
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    print(
        coverage_df.to_string(
            index=False
        )
    )

    coverage_df.to_csv(
        os.path.join(
            output_dir,
            "region_year_coverage.csv",
        ),
        index=False,
    )

    common_groups = sorted(
        coverage_df.loc[
            coverage_df[
                "Both_Years"
            ],
            "Group",
        ].tolist()
    )

    print(
        f"\nRegions represented in both years: "
        f"{len(common_groups)} / "
        f"{len(coverage_df)}"
    )

    # ========================================================
    # 4. CROSS-YEAR FEASIBILITY
    # ========================================================

    print(
        "\n[4] CROSS-YEAR VALIDATION FEASIBILITY"
    )

    groups_1394 = set(
        analysis_df.loc[
            analysis_df[
                "HarvestYear"
            ] == 1394,
            "Group",
        ]
    )

    groups_1404 = set(
        analysis_df.loc[
            analysis_df[
                "HarvestYear"
            ] == 1404,
            "Group",
        ]
    )

    print(
        f"Regions in 1394 : "
        f"{len(groups_1394)}"
    )

    print(
        f"Regions in 1404 : "
        f"{len(groups_1404)}"
    )

    print(
        f"Common regions  : "
        f"{len(common_groups)}"
    )

    print(
        "Common groups   : "
        + ", ".join(
            common_groups
        )
    )

    # ========================================================
    # 5. YEAR EFFECT DIAGNOSTICS
    # ========================================================

    print(
        "\n[5] HARVEST YEAR EFFECT DIAGNOSTICS"
    )

    print(
        "Three complementary analyses are performed:"
        "\n1) PCA by year"
        "\n2) Cross-validated year prediction"
        "\n3) Within-region permutation test"
    )

    # --------------------------------------------------------
    # 5A
    # --------------------------------------------------------

    pca_scores_df, _ = _run_year_pca(
        X=X,
        metadata=analysis_df,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # 5B
    # --------------------------------------------------------

    year_cv_result = _run_year_prediction_cv(
        X=X,
        metadata=analysis_df,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # 5C
    # --------------------------------------------------------

    pc_columns = [
        column
        for column in pca_scores_df.columns
        if column.startswith(
            "PC"
        )
    ]

    within_region_result = (
        _run_within_region_year_permutation(
            pca_scores=
                pca_scores_df[
                    pc_columns
                ],
            metadata=
                analysis_df,
            output_dir=
                output_dir,
            n_permutations=
                2000,
        )
    )

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    year_cv_ba = (
        year_cv_result[
            "BalancedAccuracy"
        ]
    )

    binary_chance_ba = 0.5

    within_p = (
        within_region_result[
            "PermutationPValue"
        ]
    )

    if (
        year_cv_ba
        > binary_chance_ba
        and not np.isnan(
            within_p
        )
        and within_p < 0.05
    ):

        overall_interpretation = (
            "STRONG EVIDENCE OF A "
            "HARVEST-YEAR EFFECT: "
            "cross-validated spectral "
            "prediction of year is above "
            "chance and the within-region "
            "permutation test is significant."
        )

    elif (
        year_cv_ba
        > binary_chance_ba
        or (
            not np.isnan(
                within_p
            )
            and within_p < 0.05
        )
    ):

        overall_interpretation = (
            "EVIDENCE OF A POSSIBLE "
            "HARVEST-YEAR EFFECT: "
            "at least one independent "
            "diagnostic indicates "
            "year-related spectral structure."
        )

    else:

        overall_interpretation = (
            "NO STRONG EVIDENCE OF A "
            "HARVEST-YEAR EFFECT FROM "
            "THE CURRENT DIAGNOSTICS."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "HARVEST YEAR EFFECT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Cross-validated year BA : "
        f"{year_cv_ba:.4f}"
    )

    print(
        f"Chance BA               : "
        f"{binary_chance_ba:.4f}"
    )

    if np.isnan(
        within_p
    ):

        print(
            "Within-region p-value   : NA"
        )

    else:

        print(
            f"Within-region p-value   : "
            f"{within_p:.4f}"
        )

    print(
        "\nInterpretation:"
    )

    print(
        overall_interpretation
    )

    year_summary = pd.DataFrame(
        [
            {
                "YearPredictionBalancedAccuracy":
                    year_cv_result[
                        "BalancedAccuracy"
                    ],
                "YearPredictionMacroF1":
                    year_cv_result[
                        "F1Macro"
                    ],
                "YearPredictionAccuracy":
                    year_cv_result[
                        "Accuracy"
                    ],
                "ChanceBalancedAccuracy":
                    binary_chance_ba,
                "WithinRegionPermutationPValue":
                    within_p,
                "CommonRegions":
                    within_region_result[
                        "CommonRegions"
                    ],
                "SamplesIncludedWithinRegion":
                    within_region_result[
                        "SamplesIncluded"
                    ],
                "Interpretation":
                    overall_interpretation,
            }
        ]
    )

    year_summary_path = os.path.join(
        output_dir,
        "harvest_year_effect_summary.csv",
    )

    year_summary.to_csv(
        year_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nSaved: "
        f"{year_summary_path}"
    )

    # ========================================================
    # 6. CROSS-YEAR ORIGIN VALIDATION
    # ========================================================

    print(
        "\n[6] CROSS-YEAR ORIGIN VALIDATION"
    )

    print(
        "Only regions represented in BOTH years "
        "are included."
    )

    print(
        "Leakage control:"
        "\nFeature selection is fitted only on "
        "the training year."
    )

    if len(
        common_groups
    ) >= 2:

        all_results = []

        # ----------------------------------------------------
        # 1394 -> 1404
        # ----------------------------------------------------

        print(
            "\n1394 -> 1404"
        )

        results_forward = (
            _run_cross_year_direction(
                X=X,
                metadata=analysis_df,
                source_year=1394,
                target_year=1404,
                common_groups=
                    common_groups,
                output_dir=
                    output_dir,
            )
        )

        all_results.extend(
            results_forward
        )

        # ----------------------------------------------------
        # 1404 -> 1394
        # ----------------------------------------------------

        print(
            "\n1404 -> 1394"
        )

        results_reverse = (
            _run_cross_year_direction(
                X=X,
                metadata=analysis_df,
                source_year=1404,
                target_year=1394,
                common_groups=
                    common_groups,
                output_dir=
                    output_dir,
            )
        )

        all_results.extend(
            results_reverse
        )

    else:

        all_results = []

    # ========================================================
    # 7. SAVE CROSS-YEAR RESULTS
    # ========================================================

    print(
        "\n[7] CROSS-YEAR RESULTS"
    )

    if all_results:

        results_df = pd.DataFrame(
            all_results
        )

        results_df = (
            results_df.sort_values(
                [
                    "BalancedAccuracy",
                    "F1Macro",
                    "Accuracy",
                ],
                ascending=False,
            )
        )

        print(
            results_df.to_string(
                index=False
            )
        )

        results_path = os.path.join(
            output_dir,
            "cross_year_origin_results.csv",
        )

        results_df.to_csv(
            results_path,
            index=False,
            encoding="utf-8-sig",
        )

        direction_summary = (
            results_df
            .groupby(
                [
                    "SourceYear",
                    "TargetYear",
                    "Model",
                    "TopK",
                ],
                as_index=False,
            )
            .agg(
                {
                    "Accuracy":
                        "mean",
                    "BalancedAccuracy":
                        "mean",
                    "F1Macro":
                        "mean",
                }
            )
        )

        direction_summary_path = os.path.join(
            output_dir,
            "cross_year_origin_summary.csv",
        )

        direction_summary.to_csv(
            direction_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\nSaved:"
        )

        print(
            f"- {results_path}"
        )

        print(
            f"- {direction_summary_path}"
        )

        best = results_df.iloc[
            0
        ]

        print(
            "\n" + "=" * 70
        )

        print(
            "BEST CROSS-YEAR CANDIDATE"
        )

        print(
            "=" * 70
        )

        print(
            f"Direction          : "
            f"{int(best['SourceYear'])} -> "
            f"{int(best['TargetYear'])}"
        )

        print(
            f"Model              : "
            f"{best['Model']}"
        )

        print(
            f"Top-K              : "
            f"{int(best['TopK'])}"
        )

        print(
            f"Accuracy           : "
            f"{best['Accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy  : "
            f"{best['BalancedAccuracy']:.4f}"
        )

        print(
            f"Macro F1           : "
            f"{best['F1Macro']:.4f}"
        )

        print(
            "\nIMPORTANT:"
            "\nThis is an out-of-year generalization result."
            "\nIt must not be interpreted as the final "
            "operational model performance."
        )

    else:

        print(
            "No cross-year results were generated."
        )

    # ========================================================
    # 8. SAMPLE METADATA
    # ========================================================

    sample_metadata = analysis_df[
        [
            "SampleId",
            "OriginalSampleName",
            "Group",
            "HarvestYear",
        ]
    ].copy()

    sample_metadata.to_csv(
        os.path.join(
            output_dir,
            "harvest_year_sample_metadata.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "HARVEST YEAR ANALYSIS FINISHED"
    )

    print(
        "=" * 70
    )

