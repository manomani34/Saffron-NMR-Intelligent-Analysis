import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def run_pca(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    output_path: str = "pca_scores.csv",
    plot_path: str = "pca_scores.png",
    explained_variance_path: str = "pca_explained_variance.csv",
    loadings_path: str = "pca_loadings.csv",
    scree_plot_path: str = "pca_scree_plot.png",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PCA EXPLORATORY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Input information
    # --------------------------------------------------

    print("\n[1] INPUT DATA")

    print(
        f"Samples         : {X.shape[0]}"
    )

    print(
        f"Spectral points : {X.shape[1]}"
    )

    if len(X) != len(metadata):
        raise ValueError(
            "X and metadata row counts do not match."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y row counts do not match."
        )

    # --------------------------------------------------
    # 2. PCA
    # --------------------------------------------------
    #
    # PCA centers the variables internally.
    #
    # No StandardScaler is applied here because this
    # is an exploratory PCA of the NMR spectral matrix.
    #
    # Scaling may be evaluated separately later as part
    # of a controlled preprocessing experiment.
    # --------------------------------------------------

    print("\n[2] RUNNING PCA")

    pca = PCA()

    scores = pca.fit_transform(X)

    explained_variance_ratio = (
        pca.explained_variance_ratio_
    )

    cumulative_variance = (
        explained_variance_ratio.cumsum()
    )

    # --------------------------------------------------
    # 3. Explained variance
    # --------------------------------------------------

    print("\n[3] EXPLAINED VARIANCE")

    variance_rows = []

    for i in range(
        len(explained_variance_ratio)
    ):

        variance_rows.append(
            {
                "PC": i + 1,
                "ExplainedVarianceRatio":
                    explained_variance_ratio[i],
                "ExplainedVariancePercent":
                    explained_variance_ratio[i] * 100,
                "CumulativeVarianceRatio":
                    cumulative_variance[i],
                "CumulativeVariancePercent":
                    cumulative_variance[i] * 100,
            }
        )

    explained_variance_df = pd.DataFrame(
        variance_rows
    )

    for i in range(
        min(10, len(explained_variance_ratio))
    ):

        print(
            f"PC{i + 1:2d} -> "
            f"{explained_variance_ratio[i] * 100:.4f}% "
            f"(cumulative: "
            f"{cumulative_variance[i] * 100:.4f}%)"
        )

    # --------------------------------------------------
    # 4. Components for 90% / 95%
    # --------------------------------------------------

    pc_90 = next(
        (
            i + 1
            for i, value in enumerate(
                cumulative_variance
            )
            if value >= 0.90
        ),
        None,
    )

    pc_95 = next(
        (
            i + 1
            for i, value in enumerate(
                cumulative_variance
            )
            if value >= 0.95
        ),
        None,
    )

    print("\nComponents required:")

    print(
        f"90% cumulative variance : {pc_90}"
    )

    print(
        f"95% cumulative variance : {pc_95}"
    )

    # --------------------------------------------------
    # 5. Save explained variance
    # --------------------------------------------------

    explained_variance_df.to_csv(
        explained_variance_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nExplained variance saved to: "
        f"{explained_variance_path}"
    )

    # --------------------------------------------------
    # 6. Create PCA score table
    # --------------------------------------------------

    score_columns = [
        f"PC{i + 1}"
        for i in range(scores.shape[1])
    ]

    scores_df = pd.DataFrame(
        scores,
        columns=score_columns,
    )

    # Metadata
    scores_df.insert(
        0,
        "SampleId",
        metadata["SampleId"].to_numpy(),
    )

    scores_df.insert(
        1,
        "Group",
        metadata["Group"].to_numpy(),
    )

    if "HarvestYear" in metadata.columns:

        scores_df.insert(
            2,
            "HarvestYear",
            metadata["HarvestYear"].to_numpy(),
        )

    # --------------------------------------------------
    # 7. Save PCA scores
    # --------------------------------------------------

    scores_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"PCA scores saved to: {output_path}"
    )

    # --------------------------------------------------
    # 8. PCA loadings
    # --------------------------------------------------
    #
    # Components:
    #   n_components x spectral_variables
    #
    # We transpose them so that each row represents
    # one spectral variable / ppm position.
    # --------------------------------------------------

    loadings = pca.components_.T

    loading_columns = [
        f"PC{i + 1}"
        for i in range(
            pca.components_.shape[0]
        )
    ]

    loadings_df = pd.DataFrame(
        loadings,
        columns=loading_columns,
        index=X.columns,
    )

    loadings_df.index.name = "PPM"

    loadings_df.to_csv(
        loadings_path,
        encoding="utf-8-sig",
    )

    print(
        f"PCA loadings saved to: {loadings_path}"
    )

    # --------------------------------------------------
    # 9. Generate PCA score plot
    # --------------------------------------------------

    print("\n[4] GENERATING PCA SCORE PLOT")

    if scores.shape[1] < 2:

        raise ValueError(
            "At least two principal components are "
            "required for a PC1-PC2 plot."
        )

    plt.figure(
        figsize=(10, 7)
    )

    unique_groups = sorted(
        y.astype(str).unique(),
        key=lambda value:
            int(value[1:])
            if str(value).startswith("G")
            else 999,
    )

    for group in unique_groups:

        mask = (
            y.astype(str).to_numpy()
            == group
        )

        plt.scatter(
            scores[mask, 0],
            scores[mask, 1],
            label=group,
            s=70,
        )

    # --------------------------------------------------
    # Sample labels
    # --------------------------------------------------

    for index in range(
        len(metadata)
    ):

        sample_id = metadata.iloc[
            index
        ]["SampleId"]

        plt.annotate(
            str(sample_id),
            (
                scores[index, 0],
                scores[index, 1],
            ),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel(
        f"PC1 "
        f"({explained_variance_ratio[0] * 100:.2f}%)"
    )

    plt.ylabel(
        f"PC2 "
        f"({explained_variance_ratio[1] * 100:.2f}%)"
    )

    plt.title(
        "PCA of Saffron NMR Spectral Data"
    )

    plt.legend(
        title="Group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"PCA score plot saved to: {plot_path}"
    )

    # --------------------------------------------------
    # 10. Scree plot
    # --------------------------------------------------

    print(
        "\n[5] GENERATING SCREE PLOT"
    )

    plt.figure(
        figsize=(10, 6)
    )

    pc_numbers = range(
        1,
        len(explained_variance_ratio) + 1,
    )

    plt.plot(
        pc_numbers,
        explained_variance_ratio * 100,
        marker="o",
    )

    plt.xlabel(
        "Principal Component"
    )

    plt.ylabel(
        "Explained Variance (%)"
    )

    plt.title(
        "PCA Scree Plot"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        scree_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Scree plot saved to: "
        f"{scree_plot_path}"
    )

    # --------------------------------------------------
    # 11. Interpretation
    # --------------------------------------------------

    print("\n[6] INTERPRETATION")

    print(
        "PCA is exploratory and unsupervised."
    )

    print(
        "Visible group separation in PCA does not "
        "constitute classification performance."
    )

    print(
        "PCA scores and loadings will be used as "
        "supporting information for subsequent "
        "novelty/OOD analysis."
    )

    print(
        "No accuracy claim is made from PCA."
    )

    # --------------------------------------------------
    # 12. Final summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("PCA ANALYSIS FINISHED")
    print("=" * 70)

    print(
        f"Scores shape       : {scores_df.shape}"
    )

    print(
        f"Loadings shape     : {loadings_df.shape}"
    )

    print(
        f"90% variance PCs   : {pc_90}"
    )

    print(
        f"95% variance PCs   : {pc_95}"
    )

    return scores_df