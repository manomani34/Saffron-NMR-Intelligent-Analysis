from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    "reports/color_pca_color_region"
)

COLOR_REGION = (
    5.0,
    9.0,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR REGION PCA (5–9 ppm)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X = np.load(
        X_PATH
    )

    ppm = np.load(
        PPM_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    print()
    print(
        f"Original X : {X.shape}"
    )

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    mask = (
        (ppm >= COLOR_REGION[0])
        & (ppm <= COLOR_REGION[1])
    )

    X_region = X[
        :,
        mask,
    ]

    ppm_region = ppm[
        mask
    ]

    print(
        f"Region : "
        f"{COLOR_REGION[0]} - "
        f"{COLOR_REGION[1]} ppm"
    )

    print(
        f"Region X : "
        f"{X_region.shape}"
    )

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    print()
    print("[1] STANDARDIZATION")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_region
    )

    print(
        f"Scaled X : "
        f"{X_scaled.shape}"
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print()
    print("[2] PCA")

    pca = PCA()

    scores = pca.fit_transform(
        X_scaled
    )

    explained = (
        pca.explained_variance_ratio_
    )

    print(
        f"PC1 explained variance : "
        f"{explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance : "
        f"{explained[1] * 100:.2f}%"
    )

    if len(explained) >= 3:

        print(
            f"PC3 explained variance : "
            f"{explained[2] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    scores_df = pd.DataFrame(
        {
            "number":
                meta["number"],

            "name":
                meta["name"],

            "class":
                meta["class"],

            "PC1":
                scores[:, 0],

            "PC2":
                scores[:, 1],
        }
    )

    if scores.shape[1] >= 3:

        scores_df["PC3"] = (
            scores[:, 2]
        )

    scores_path = (
        OUTPUT_DIR
        / "color_region_pca_scores.csv"
    )

    scores_df.to_csv(
        scores_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # EXPLAINED VARIANCE
    # --------------------------------------------------------

    variance_df = pd.DataFrame(
        {
            "PC":
                np.arange(
                    1,
                    len(explained) + 1,
                ),

            "explained_percent":
                explained * 100,

            "cumulative_percent":
                np.cumsum(
                    explained
                ) * 100,
        }
    )

    variance_path = (
        OUTPUT_DIR
        / "color_region_explained_variance.csv"
    )

    variance_df.to_csv(
        variance_path,
        index=False,
    )

    # --------------------------------------------------------
    # PCA PLOT
    # --------------------------------------------------------

    print()
    print("[3] PC1 vs PC2")

    plt.figure(
        figsize=(12, 9)
    )

    classes = (
        meta["class"]
        .astype(str)
        .unique()
    )

    for class_name in classes:

        class_mask = (
            meta["class"]
            == class_name
        )

        indexes = np.where(
            class_mask.to_numpy()
        )[0]

        plt.scatter(
            scores[
                indexes,
                0
            ],
            scores[
                indexes,
                1
            ],
            s=80,
            alpha=0.85,
            label=class_name,
        )

        for index in indexes:

            plt.annotate(
                str(
                    meta.iloc[index]["number"]
                ),
                (
                    scores[index, 0],
                    scores[index, 1],
                ),
                fontsize=8,
            )

    plt.xlabel(
        f"PC1 ({explained[0] * 100:.2f}%)"
    )

    plt.ylabel(
        f"PC2 ({explained[1] * 100:.2f}%)"
    )

    plt.title(
        "PCA of Artificial Color Region (5–9 ppm)"
    )

    plt.legend(
        fontsize=8,
        loc="best",
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "color_region_pca_pc1_pc2.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # PC1 LOADING
    # --------------------------------------------------------

    print()
    print("[4] PC1 LOADING")

    loading = (
        pca.components_[0]
    )

    loading_df = pd.DataFrame(
        {
            "ppm":
                ppm_region,

            "PC1_loading":
                loading,
        }
    )

    loading_path = (
        OUTPUT_DIR
        / "pc1_loadings.csv"
    )

    loading_df.to_csv(
        loading_path,
        index=False,
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        ppm_region,
        loading,
        linewidth=1.0,
    )

    plt.xlabel(
        "Chemical shift (ppm)"
    )

    plt.ylabel(
        "PC1 loading"
    )

    plt.title(
        "PC1 Loading - Artificial Color Region"
    )

    plt.gca().invert_xaxis()

    plt.tight_layout()

    loading_plot_path = (
        OUTPUT_DIR
        / "pc1_loadings.png"
    )

    plt.savefig(
        loading_plot_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLOR REGION PCA COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Scores          : {scores_path}"
    )

    print(
        f"Variance        : {variance_path}"
    )

    print(
        f"PCA plot        : {plot_path}"
    )

    print(
        f"PC1 loadings    : {loading_path}"
    )

    print(
        f"Loading plot    : {loading_plot_path}"
    )


if __name__ == "__main__":
    main()