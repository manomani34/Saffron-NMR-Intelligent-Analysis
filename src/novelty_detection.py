# src/novelty_detection.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2
from sklearn.covariance import LedoitWolf


def run_novelty_detection(
    pca_scores_path: str = "pca_scores.csv",
    explained_variance_path: str = "pca_explained_variance.csv",
    output_path: str = "novelty_detection_results.csv",
    plot_path: str = "novelty_detection.png",
    confidence_level: float = 0.99,
    target_variance: float = 0.95,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("MAHALANOBIS NOVELTY / OOD ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load PCA files
    # --------------------------------------------------

    pca_path = Path(pca_scores_path)
    variance_path = Path(explained_variance_path)

    if not pca_path.exists():
        raise FileNotFoundError(
            f"PCA scores file not found: {pca_scores_path}"
        )

    if not variance_path.exists():
        raise FileNotFoundError(
            f"Explained variance file not found: "
            f"{explained_variance_path}"
        )

    scores_df = pd.read_csv(
        pca_path,
        encoding="utf-8-sig",
    )

    variance_df = pd.read_csv(
        variance_path,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 2. Detect PCA score columns
    # --------------------------------------------------

    pc_columns = [
        column
        for column in scores_df.columns
        if str(column).strip().upper().startswith("PC")
    ]

    if not pc_columns:
        raise ValueError(
            "No PCA score columns found in pca_scores.csv."
        )

    # Sort PC columns numerically
    def pc_number(column):
        text = str(column).strip().upper()

        try:
            return int(text.replace("PC", ""))
        except ValueError:
            return 999999

    pc_columns = sorted(
        pc_columns,
        key=pc_number,
    )

    # --------------------------------------------------
    # 3. Determine PCs required for target variance
    # --------------------------------------------------

    cumulative_values = None

    # Prefer exact known column from our PCA output
    if "CumulativeVarianceRatio" in variance_df.columns:

        cumulative_values = pd.to_numeric(
            variance_df["CumulativeVarianceRatio"],
            errors="coerce",
        )

    elif "CumulativeVariancePercent" in variance_df.columns:

        cumulative_values = (
            pd.to_numeric(
                variance_df[
                    "CumulativeVariancePercent"
                ],
                errors="coerce",
            )
            / 100.0
        )

    else:

        # Generic fallback
        cumulative_column = None

        for column in variance_df.columns:

            normalized = (
                str(column)
                .strip()
                .lower()
            )

            if (
                "cumulative" in normalized
                and "variance" in normalized
            ):
                cumulative_column = column
                break

        if cumulative_column is not None:

            cumulative_values = pd.to_numeric(
                variance_df[cumulative_column],
                errors="coerce",
            )

            cumulative_values = (
                cumulative_values.dropna()
            )

            if not cumulative_values.empty:

                if cumulative_values.max() > 1.0:
                    cumulative_values = (
                        cumulative_values / 100.0
                    )

        else:

            explained_column = None

            for column in variance_df.columns:

                normalized = (
                    str(column)
                    .strip()
                    .lower()
                )

                if (
                    "explained" in normalized
                    and "variance" in normalized
                ):
                    explained_column = column
                    break

            if explained_column is None:
                raise ValueError(
                    "Could not identify PCA variance columns."
                )

            explained_values = pd.to_numeric(
                variance_df[explained_column],
                errors="coerce",
            ).dropna()

            if explained_values.empty:
                raise ValueError(
                    "Explained variance data is empty."
                )

            if explained_values.max() > 1.0:
                explained_values = (
                    explained_values / 100.0
                )

            cumulative_values = (
                explained_values.cumsum()
            )

    cumulative_values = pd.Series(
        cumulative_values
    ).dropna()

    if cumulative_values.empty:
        raise ValueError(
            "Cumulative PCA variance data is empty."
        )

    cumulative_array = (
        cumulative_values.to_numpy(dtype=float)
    )

    # Find first PC reaching target variance
    matching_indices = np.where(
        cumulative_array >= target_variance
    )[0]

    if len(matching_indices) == 0:

        components_required = len(
            cumulative_array
        )

    else:

        components_required = (
            int(matching_indices[0]) + 1
        )

    # --------------------------------------------------
    # 4. Limit number of PCs
    # --------------------------------------------------

    # Mahalanobis covariance estimation cannot use
    # as many dimensions as observations.
    max_allowed_components = min(
        len(pc_columns),
        scores_df.shape[0] - 1,
    )

    n_components = min(
        components_required,
        max_allowed_components,
    )

    if n_components < 2:
        raise ValueError(
            "At least 2 PCA components are required "
            "for novelty analysis."
        )

    selected_pc_columns = pc_columns[
        :n_components
    ]

    actual_cumulative_variance = float(
        cumulative_array[n_components - 1]
    )

    print(
        f"\nTarget cumulative variance : "
        f"{target_variance * 100:.1f}%"
    )

    print(
        f"PCA components required    : "
        f"{components_required}"
    )

    print(
        f"PCA components used        : "
        f"{n_components}"
    )

    print(
        f"Achieved cumulative variance: "
        f"{actual_cumulative_variance * 100:.4f}%"
    )

    print(
        f"Selected PCs               : "
        f"{', '.join(selected_pc_columns)}"
    )

    # --------------------------------------------------
    # 5. Prepare PCA matrix
    # --------------------------------------------------

    X_pca = (
        scores_df[
            selected_pc_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if X_pca.isna().any().any():
        raise ValueError(
            "PCA score matrix contains invalid values."
        )

    X = X_pca.to_numpy(
        dtype=float
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "PCA score matrix contains NaN "
            "or infinite values."
        )

    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            "Number of observations must be greater "
            "than number of PCA components."
        )

    # --------------------------------------------------
    # 6. Mahalanobis covariance model
    # --------------------------------------------------

    print(
        "\n[1] FITTING COVARIANCE MODEL"
    )

    covariance_model = LedoitWolf(
        assume_centered=False,
    )

    covariance_model.fit(X)

    mean_vector = (
        covariance_model.location_
    )

    precision_matrix = (
        covariance_model.precision_
    )

    centered = (
        X - mean_vector
    )

    mahalanobis_squared = np.einsum(
        "ij,jk,ik->i",
        centered,
        precision_matrix,
        centered,
    )

    mahalanobis_squared = np.maximum(
        mahalanobis_squared,
        0.0,
    )

    mahalanobis_distance = np.sqrt(
        mahalanobis_squared
    )

    # --------------------------------------------------
    # 7. Chi-square threshold
    # --------------------------------------------------

    threshold_squared = chi2.ppf(
        confidence_level,
        df=n_components,
    )

    threshold_distance = np.sqrt(
        threshold_squared
    )

    is_novel = (
        mahalanobis_squared
        > threshold_squared
    )

    print(
        f"Confidence level     : "
        f"{confidence_level:.3f}"
    )

    print(
        f"Degrees of freedom   : "
        f"{n_components}"
    )

    print(
        f"Threshold (D²)       : "
        f"{threshold_squared:.4f}"
    )

    print(
        f"Threshold (D)        : "
        f"{threshold_distance:.4f}"
    )

    print(
        f"Novel observations   : "
        f"{int(is_novel.sum())}"
    )

    print(
        f"Within reference set : "
        f"{int((~is_novel).sum())}"
    )

    # --------------------------------------------------
    # 8. Build result table
    # --------------------------------------------------

    if "SampleId" in scores_df.columns:

        sample_ids = (
            scores_df["SampleId"]
        )

    else:

        sample_ids = pd.Series(
            np.arange(
                1,
                len(scores_df) + 1,
            )
        )

    if "Group" in scores_df.columns:

        groups = (
            scores_df["Group"]
        )

    else:

        groups = pd.Series(
            [""] * len(scores_df)
        )

    results = pd.DataFrame(
        {
            "SampleId": sample_ids,
            "Group": groups,
            "MahalanobisSquared":
                mahalanobis_squared,
            "MahalanobisDistance":
                mahalanobis_distance,
            "ThresholdSquared":
                threshold_squared,
            "ThresholdDistance":
                threshold_distance,
            "IsNovel":
                is_novel,
            "NoveltyStatus":
                np.where(
                    is_novel,
                    "NOVEL / OOD CANDIDATE",
                    "WITHIN REFERENCE DOMAIN",
                ),
        }
    )

    results = results.sort_values(
        by="MahalanobisSquared",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # 9. Show novel observations
    # --------------------------------------------------

    print(
        "\n[2] NOVEL OBSERVATIONS"
    )

    novel_results = results[
        results["IsNovel"]
    ]

    if novel_results.empty:

        print(
            "No novel observations detected."
        )

    else:

        print(
            novel_results[
                [
                    "SampleId",
                    "Group",
                    "MahalanobisSquared",
                    "MahalanobisDistance",
                    "NoveltyStatus",
                ]
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    # --------------------------------------------------
    # 11. Plot
    # --------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    x_axis = np.arange(
        1,
        len(results) + 1,
    )

    plt.plot(
        x_axis,
        results["MahalanobisSquared"],
        marker="o",
        linewidth=1,
    )

    plt.axhline(
        threshold_squared,
        linestyle="--",
        linewidth=2,
        label=(
            f"{int(confidence_level * 100)}% "
            f"threshold = "
            f"{threshold_squared:.2f}"
        ),
    )

    novel_positions = (
        np.where(
            results["IsNovel"].to_numpy()
        )[0] + 1
    )

    if len(novel_positions) > 0:

        novel_values = results.loc[
            results["IsNovel"],
            "MahalanobisSquared",
        ]

        plt.scatter(
            novel_positions,
            novel_values,
            s=60,
            label="Novel / OOD candidate",
            zorder=3,
        )

    plt.xlabel(
        "Samples sorted by Mahalanobis distance"
    )

    plt.ylabel(
        "Mahalanobis Distance²"
    )

    plt.title(
        "Mahalanobis Novelty / OOD Detection"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Plot saved to: {plot_path}"
    )

    # --------------------------------------------------
    # 12. Final
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "MAHALANOBIS NOVELTY / OOD ANALYSIS FINISHED"
    )
    print("=" * 70)

    return results