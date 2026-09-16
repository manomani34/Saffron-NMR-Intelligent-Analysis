from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    "reports/color_exploratory"
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
# NORMALIZATION
# ============================================================

def snv(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)

    std[std == 0] = 1.0

    return (X - mean) / std


# ============================================================
# EFFECT PROFILE
# ============================================================

def calculate_effect_profile(
    X,
    meta,
    reference_mask,
    target_mask,
    ppm,
):
    reference = X[
        reference_mask
    ]

    target = X[
        target_mask
    ]

    reference_mean = reference.mean(
        axis=0
    )

    target_mean = target.mean(
        axis=0
    )

    reference_std = reference.std(
        axis=0
    )

    target_std = target.std(
        axis=0
    )

    pooled_std = np.sqrt(
        (
            reference_std ** 2
            + target_std ** 2
        ) / 2.0
    )

    pooled_std[
        pooled_std == 0
    ] = 1.0

    effect = (
        target_mean
        - reference_mean
    ) / pooled_std

    result = pd.DataFrame(
        {
            "ppm": ppm,
            "reference_mean":
                reference_mean,
            "target_mean":
                target_mean,
            "mean_difference":
                target_mean
                - reference_mean,
            "standardized_effect":
                effect,
            "absolute_effect":
                np.abs(effect),
        }
    )

    return result.sort_values(
        "absolute_effect",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR EXPLORATORY ANALYSIS")
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
    # 5–9 PPM
    # --------------------------------------------------------

    region_mask = (
        (ppm >= COLOR_REGION[0])
        & (ppm <= COLOR_REGION[1])
    )

    X_region = X[
        :,
        region_mask,
    ]

    ppm_region = ppm[
        region_mask
    ]

    print(
        f"Color region : "
        f"{COLOR_REGION[0]} - "
        f"{COLOR_REGION[1]} ppm"
    )

    print(
        f"Region X : "
        f"{X_region.shape}"
    )

    # --------------------------------------------------------
    # SNV
    # --------------------------------------------------------

    print()
    print("[1] SNV NORMALIZATION")

    X_snv = snv(
        X_region
    )

    print(
        f"SNV matrix : "
        f"{X_snv.shape}"
    )

    np.save(
        OUTPUT_DIR
        / "color_region_snv.npy",
        X_snv,
    )

    # --------------------------------------------------------
    # SAMPLE GROUPS
    # --------------------------------------------------------

    real_mask = (
        meta["class"]
        == "real_saffron"
    ).to_numpy()

    pure_color_mask = (
        meta["class"]
        == "pure_artificial_color"
    ).to_numpy()

    mix_5_mask = (
        (
            meta["class"]
            == "saffron_plus_artificial_color"
        )
        & (
            meta[
                "artificial_color_percent"
            ]
            == 5.0
        )
    ).to_numpy()

    mix_10_mask = (
        (
            meta["class"]
            == "saffron_plus_artificial_color"
        )
        & (
            meta[
                "artificial_color_percent"
            ]
            == 10.0
        )
    ).to_numpy()

    known_adulterated_mask = (
        meta["known_artificial_color"]
        .astype(bool)
        .to_numpy()
        & (
            meta["class"]
            != "pure_artificial_color"
        )
        & (
            meta["class"]
            != "saffron_plus_artificial_color"
        )
    )

    unknown_mask = (
        meta["unknown_adulteration"]
        .astype(bool)
        .to_numpy()
    )

    print()
    print("[2] SAMPLE GROUPS")

    print(
        f"Real saffron                 : "
        f"{real_mask.sum()}"
    )

    print(
        f"Pure artificial colors       : "
        f"{pure_color_mask.sum()}"
    )

    print(
        f"Saffron + color (5%)         : "
        f"{mix_5_mask.sum()}"
    )

    print(
        f"Saffron + color (10%)        : "
        f"{mix_10_mask.sum()}"
    )

    print(
        f"Known artificial adulterated : "
        f"{known_adulterated_mask.sum()}"
    )

    print(
        f"Unknown adulteration         : "
        f"{unknown_mask.sum()}"
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print()
    print("[3] PCA ON SNV DATA")

    X_scaled = StandardScaler().fit_transform(
        X_snv
    )

    n_components = min(
        10,
        X_scaled.shape[0] - 1,
        X_scaled.shape[1],
    )

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(
        X_scaled
    )

    explained = (
        pca.explained_variance_ratio_
    )

    scores_df = pd.DataFrame(
        {
            "number":
                meta["number"],

            "name":
                meta["name"],

            "class":
                meta["class"],

            "artificial_color":
                meta["artificial_color"],
        }
    )

    for i in range(
        scores.shape[1]
    ):
        scores_df[
            f"PC{i + 1}"
        ] = scores[:, i]

    scores_path = (
        OUTPUT_DIR
        / "snv_pca_scores.csv"
    )

    scores_df.to_csv(
        scores_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"PC1 : {explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 : {explained[1] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PCA PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 9)
    )

    classes = [
        "real_saffron",
        "pure_artificial_color",
        "saffron_plus_artificial_color",
        "artificial_color_adulterated_saffron",
        "small_amount_artificial_color",
        "unknown_adulterated_saffron",
        "artificial_saffron",
    ]

    for class_name in classes:

        mask = (
            meta["class"]
            == class_name
        ).to_numpy()

        if not mask.any():
            continue

        plt.scatter(
            scores[
                mask,
                0
            ],
            scores[
                mask,
                1
            ],
            s=85,
            alpha=0.8,
            label=class_name,
        )

        for index in np.where(mask)[0]:

            plt.annotate(
                str(
                    meta.iloc[index]["number"]
                ),
                (
                    scores[
                        index,
                        0
                    ],
                    scores[
                        index,
                        1
                    ],
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
        "SNV-PCA: Artificial Color Region (5–9 ppm)"
    )

    plt.legend(
        fontsize=8,
        loc="best",
    )

    plt.tight_layout()

    pca_plot = (
        OUTPUT_DIR
        / "snv_pca.png"
    )

    plt.savefig(
        pca_plot,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # REAL SAFFRON VS 5% MIXTURES
    # --------------------------------------------------------

    print()
    print("[4] REAL SAFFRON VS 5% MIXTURES")

    if (
        real_mask.sum() >= 1
        and mix_5_mask.sum() >= 1
    ):

        effect_5 = calculate_effect_profile(
            X_snv,
            meta,
            real_mask,
            mix_5_mask,
            ppm_region,
        )

        effect_5_path = (
            OUTPUT_DIR
            / "real_vs_5pct_effect.csv"
        )

        effect_5.to_csv(
            effect_5_path,
            index=False,
        )

        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            effect_5["ppm"],
            effect_5["standardized_effect"],
            linewidth=1.0,
        )

        plt.axhline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "Chemical shift (ppm)"
        )

        plt.ylabel(
            "Standardized effect"
        )

        plt.title(
            "Real Saffron vs 5% Artificial-Color Mixtures"
        )

        plt.gca().invert_xaxis()

        plt.tight_layout()

        effect_plot = (
            OUTPUT_DIR
            / "real_vs_5pct_effect.png"
        )

        plt.savefig(
            effect_plot,
            dpi=200,
        )

        plt.close()

        print(
            f"Top effect: "
            f"{effect_5.iloc[0]['ppm']:.6f} ppm"
        )

    # --------------------------------------------------------
    # REAL SAFFRON VS 10% MIXTURES
    # --------------------------------------------------------

    print()
    print("[5] REAL SAFFRON VS 10% MIXTURES")

    if (
        real_mask.sum() >= 1
        and mix_10_mask.sum() >= 1
    ):

        effect_10 = calculate_effect_profile(
            X_snv,
            meta,
            real_mask,
            mix_10_mask,
            ppm_region,
        )

        effect_10_path = (
            OUTPUT_DIR
            / "real_vs_10pct_effect.csv"
        )

        effect_10.to_csv(
            effect_10_path,
            index=False,
        )

        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            effect_10["ppm"],
            effect_10["standardized_effect"],
            linewidth=1.0,
        )

        plt.axhline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "Chemical shift (ppm)"
        )

        plt.ylabel(
            "Standardized effect"
        )

        plt.title(
            "Real Saffron vs 10% Artificial-Color Mixtures"
        )

        plt.gca().invert_xaxis()

        plt.tight_layout()

        effect_plot = (
            OUTPUT_DIR
            / "real_vs_10pct_effect.png"
        )

        plt.savefig(
            effect_plot,
            dpi=200,
        )

        plt.close()

        print(
            f"Top effect: "
            f"{effect_10.iloc[0]['ppm']:.6f} ppm"
        )

    # --------------------------------------------------------
    # PURE COLORS VS REAL SAFFRON
    # --------------------------------------------------------

    print()
    print("[6] PURE COLORS VS REAL SAFFRON")

    if (
        real_mask.sum() >= 1
        and pure_color_mask.sum() >= 1
    ):

        pure_effect = calculate_effect_profile(
            X_snv,
            meta,
            real_mask,
            pure_color_mask,
            ppm_region,
        )

        pure_effect_path = (
            OUTPUT_DIR
            / "real_vs_pure_color_effect.csv"
        )

        pure_effect.to_csv(
            pure_effect_path,
            index=False,
        )

        print(
            "Top discriminating ppm:"
        )

        for _, row in pure_effect.head(
            15
        ).iterrows():

            print(
                f"  "
                f"{row['ppm']:.6f} ppm  "
                f"effect="
                f"{row['standardized_effect']:.4f}"
            )

    # --------------------------------------------------------
    # SAMPLE DISTANCE TO REAL SAFFRON
    # --------------------------------------------------------

    print()
    print("[7] DISTANCE TO REAL SAFFRON")

    real_reference = X_snv[
        real_mask
    ].mean(
        axis=0
    )

    distances = np.linalg.norm(
        X_snv
        - real_reference,
        axis=1,
    )

    distance_df = pd.DataFrame(
        {
            "number":
                meta["number"],

            "name":
                meta["name"],

            "class":
                meta["class"],

            "artificial_color":
                meta["artificial_color"],

            "distance_to_real_saffron":
                distances,
        }
    )

    distance_df = (
        distance_df
        .sort_values(
            "distance_to_real_saffron",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    distance_path = (
        OUTPUT_DIR
        / "distance_to_real_saffron.csv"
    )

    distance_df.to_csv(
        distance_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # DASHBOARD RESULT
    # --------------------------------------------------------

    dashboard = {
        "analysis": {
            "name":
                "Color Exploratory Analysis",

            "region": {
                "lower_ppm":
                    COLOR_REGION[0],

                "upper_ppm":
                    COLOR_REGION[1],
            },

            "method":
                "SNV + PCA + effect profiling",

            "purpose":
                "Exploratory signal assessment",

            "predictive_model":
                False,
        },

        "samples": {
            "total":
                int(len(meta)),

            "real_saffron":
                int(real_mask.sum()),

            "pure_artificial_color":
                int(pure_color_mask.sum()),

            "saffron_plus_color_5_percent":
                int(mix_5_mask.sum()),

            "saffron_plus_color_10_percent":
                int(mix_10_mask.sum()),

            "known_artificial_adulteration":
                int(known_adulterated_mask.sum()),

            "unknown_adulteration":
                int(unknown_mask.sum()),
        },

        "pca": {
            "pc1_explained_percent":
                float(
                    explained[0] * 100
                ),

            "pc2_explained_percent":
                float(
                    explained[1] * 100
                ),

            "components":
                int(n_components),
        },

        "limitations": [
            "Only two confirmed real-saffron reference samples are available.",
            "This analysis is exploratory and is not a validated classifier.",
            "Unknown adulteration samples are kept separate from known artificial-color samples.",
        ],

        "artifacts": {
            "snv_pca_scores":
                str(scores_path),

            "snv_pca_plot":
                str(pca_plot),

            "distance_to_real_saffron":
                str(distance_path),
        },

        "dashboard_ready":
            True,
    }

    dashboard_path = (
        OUTPUT_DIR
        / "dashboard_result.json"
    )

    with open(
        dashboard_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dashboard,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPLORATORY ANALYSIS COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Dashboard JSON : "
        f"{dashboard_path}"
    )


if __name__ == "__main__":
    main()