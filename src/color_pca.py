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

META_PATH = Path(
    "data/processed/color_preprocessed.csv"
)

OUTPUT_DIR = Path(
    "reports/color_pca"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR DATASET PCA")
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

    meta = pd.read_csv(
        META_PATH
    )

    print()
    print(
        f"X shape      : {X.shape}"
    )

    print(
        f"Metadata rows: {len(meta)}"
    )

    if X.shape[0] != len(meta):
        raise ValueError(
            "X and metadata row counts do not match."
        )

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    print()
    print("[1] STANDARDIZATION")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    print(
        f"Scaled matrix : {X_scaled.shape}"
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

    cumulative = np.cumsum(
        explained
    )

    print(
        f"PC1 explained variance : "
        f"{explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance : "
        f"{explained[1] * 100:.2f}%"
    )

    print(
        f"PC3 explained variance : "
        f"{explained[2] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SCORE TABLE
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

            "PC3":
                scores[:, 2],
        }
    )

    scores_path = (
        OUTPUT_DIR
        / "pca_scores.csv"
    )

    scores_df.to_csv(
        scores_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Scores saved : {scores_path}"
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

            "explained_variance":
                explained,

            "explained_percent":
                explained * 100,

            "cumulative_percent":
                cumulative * 100,
        }
    )

    variance_path = (
        OUTPUT_DIR
        / "explained_variance.csv"
    )

    variance_df.to_csv(
        variance_path,
        index=False,
    )

    # --------------------------------------------------------
    # PC1 vs PC2
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

        mask = (
            meta["class"]
            == class_name
        )

        plt.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=70,
            alpha=0.85,
            label=class_name,
        )

        for index in np.where(
            mask.to_numpy()
        )[0]:

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
        "Color Dataset PCA"
    )

    plt.legend(
        fontsize=8,
        loc="best",
    )

    plt.tight_layout()

    pca_plot_path = (
        OUTPUT_DIR
        / "pca_pc1_pc2.png"
    )

    plt.savefig(
        pca_plot_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # EXPLAINED VARIANCE PLOT
    # --------------------------------------------------------

    print()
    print("[4] Explained Variance")

    plt.figure(
        figsize=(12, 6)
    )

    pcs = np.arange(
        1,
        len(explained) + 1,
    )

    plt.plot(
        pcs,
        explained * 100,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        "Principal Component"
    )

    plt.ylabel(
        "Explained Variance (%)"
    )

    plt.title(
        "PCA Explained Variance"
    )

    plt.tight_layout()

    variance_plot_path = (
        OUTPUT_DIR
        / "explained_variance.png"
    )

    plt.savefig(
        variance_plot_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PCA COMPLETED")
    print("=" * 70)

    print()
    print(
        f"PCA plot : {pca_plot_path}"
    )

    print(
        f"Variance : {variance_plot_path}"
    )


if __name__ == "__main__":
    main()