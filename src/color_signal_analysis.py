from pathlib import Path
import json

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
    "reports/color_signal_analysis"
)

COLOR_REGION = (
    5.0,
    9.0,
)

RANDOM_STATE = 42
N_PERMUTATIONS = 2000


# ============================================================
# HELPERS
# ============================================================

def load_data():

    X = np.load(
        X_PATH
    )

    ppm = np.load(
        PPM_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    return X, ppm, meta


def calculate_centroid_distance(
    X,
    labels,
):
    classes = np.unique(labels)

    if len(classes) != 2:
        raise ValueError(
            "Centroid distance requires exactly two classes."
        )

    class_a = classes[0]
    class_b = classes[1]

    centroid_a = X[
        labels == class_a
    ].mean(axis=0)

    centroid_b = X[
        labels == class_b
    ].mean(axis=0)

    distance = np.linalg.norm(
        centroid_a - centroid_b
    )

    return float(distance)


def permutation_test(
    X,
    labels,
    n_permutations,
    random_state,
):
    rng = np.random.default_rng(
        random_state
    )

    observed = calculate_centroid_distance(
        X,
        labels,
    )

    null_distances = np.empty(
        n_permutations,
        dtype=float,
    )

    shuffled = labels.copy()

    for i in range(n_permutations):

        rng.shuffle(
            shuffled
        )

        null_distances[i] = (
            calculate_centroid_distance(
                X,
                shuffled,
            )
        )

    p_value = (
        np.sum(
            null_distances >= observed
        )
        + 1
    ) / (
        n_permutations
        + 1
    )

    return (
        observed,
        null_distances,
        float(p_value),
    )


def find_top_discriminating_regions(
    X,
    ppm,
    labels,
    top_n=20,
):
    classes = np.unique(labels)

    if len(classes) != 2:
        return pd.DataFrame()

    class_a = classes[0]
    class_b = classes[1]

    mean_a = X[
        labels == class_a
    ].mean(axis=0)

    mean_b = X[
        labels == class_b
    ].mean(axis=0)

    std_all = X.std(
        axis=0
    )

    effect = np.abs(
        mean_a - mean_b
    ) / (
        std_all + 1e-12
    )

    result = pd.DataFrame(
        {
            "ppm": ppm,
            "mean_class_1": mean_a,
            "mean_class_2": mean_b,
            "absolute_mean_difference":
                np.abs(
                    mean_a - mean_b
                ),
            "standardized_effect":
                effect,
        }
    )

    return (
        result
        .sort_values(
            "standardized_effect",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR SIGNAL ANALYSIS")
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
    # COHORT DEFINITION
    # --------------------------------------------------------
    #
    # For a clean diagnostic we compare:
    #
    #   NEGATIVE = confirmed real saffron
    #   POSITIVE = samples with known artificial color
    #
    # Unknown adulteration is excluded from this specific test.
    #

    positive_mask = (
        meta["known_artificial_color"]
        .astype(bool)
    )

    negative_mask = (
        meta["class"]
        == "real_saffron"
    )

    analysis_mask = (
        positive_mask
        | negative_mask
    )

    X_analysis = X_region[
        analysis_mask.to_numpy()
    ]

    meta_analysis = (
        meta.loc[
            analysis_mask
        ]
        .reset_index(drop=True)
    )

    labels = (
        positive_mask[
            analysis_mask
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    n_negative = int(
        np.sum(labels == 0)
    )

    n_positive = int(
        np.sum(labels == 1)
    )

    print()
    print("[1] ANALYSIS COHORT")

    print(
        f"Confirmed real saffron : "
        f"{n_negative}"
    )

    print(
        f"Known artificial color : "
        f"{n_positive}"
    )

    print(
        "Unknown adulteration and "
        "other ambiguous classes excluded."
    )

    if n_negative < 2:
        raise ValueError(
            "Not enough confirmed real saffron samples."
        )

    if n_positive < 2:
        raise ValueError(
            "Not enough artificial-color samples."
        )

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    print()
    print("[2] STANDARDIZATION")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_analysis
    )

    # --------------------------------------------------------
    # PCA FOR DISTANCE SPACE
    # --------------------------------------------------------

    print()
    print("[3] PCA")

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

    cumulative = np.cumsum(
        explained
    )

    print(
        f"Components : "
        f"{n_components}"
    )

    print(
        f"Cumulative explained variance : "
        f"{cumulative[-1] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CENTROID DISTANCE
    # --------------------------------------------------------

    print()
    print("[4] CENTROID DISTANCE")

    observed_distance = (
        calculate_centroid_distance(
            scores,
            labels,
        )
    )

    print(
        f"Observed centroid distance : "
        f"{observed_distance:.6f}"
    )

    # --------------------------------------------------------
    # PERMUTATION TEST
    # --------------------------------------------------------

    print()
    print("[5] PERMUTATION TEST")

    print(
        f"Permutations : "
        f"{N_PERMUTATIONS}"
    )

    (
        observed,
        null_distances,
        p_value,
    ) = permutation_test(
        scores,
        labels,
        N_PERMUTATIONS,
        RANDOM_STATE,
    )

    print(
        f"Observed statistic : "
        f"{observed:.6f}"
    )

    print(
        f"Permutation p-value : "
        f"{p_value:.6f}"
    )

    # --------------------------------------------------------
    # EFFECT SIZE
    # --------------------------------------------------------

    null_mean = float(
        np.mean(
            null_distances
        )
    )

    null_std = float(
        np.std(
            null_distances
        )
    )

    if null_std > 0:
        z_score = (
            observed - null_mean
        ) / null_std
    else:
        z_score = np.nan

    print(
        f"Null mean : "
        f"{null_mean:.6f}"
    )

    print(
        f"Null std  : "
        f"{null_std:.6f}"
    )

    print(
        f"Permutation z-score : "
        f"{z_score:.4f}"
    )

    # --------------------------------------------------------
    # TOP SPECTRAL FEATURES
    # --------------------------------------------------------

    print()
    print("[6] TOP DISCRIMINATING PPM")

    top_features = (
        find_top_discriminating_regions(
            X_analysis,
            ppm_region,
            labels,
            top_n=50,
        )
    )

    top_features_path = (
        OUTPUT_DIR
        / "top_discriminating_ppm.csv"
    )

    top_features.to_csv(
        top_features_path,
        index=False,
    )

    for _, row in top_features.head(10).iterrows():

        print(
            f"  {row['ppm']:.6f} ppm  "
            f"effect={row['standardized_effect']:.4f}"
        )

    # --------------------------------------------------------
    # PCA SCORES
    # --------------------------------------------------------

    scores_data = {
        "number":
            meta_analysis["number"]
            .to_numpy(),

        "name":
            meta_analysis["name"]
            .to_numpy(),

        "class":
            meta_analysis["class"]
            .to_numpy(),

        "artificial_color":
            labels,
    }

    for i in range(
        scores.shape[1]
    ):

        scores_data[
            f"PC{i + 1}"
        ] = scores[
            :,
            i
        ]

    scores_df = pd.DataFrame(
        scores_data
    )

    scores_path = (
        OUTPUT_DIR
        / "signal_pca_scores.csv"
    )

    scores_df.to_csv(
        scores_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # PCA PLOT
    # --------------------------------------------------------

    print()
    print("[7] PCA PLOT")

    plt.figure(
        figsize=(11, 8)
    )

    for value, title in [
        (0, "Real saffron"),
        (1, "Known artificial color"),
    ]:

        mask = (
            labels == value
        )

        plt.scatter(
            scores[
                mask,
                0
            ],
            scores[
                mask,
                1
            ],
            s=80,
            alpha=0.85,
            label=title,
        )

        indexes = np.where(
            mask
        )[0]

        for index in indexes:

            plt.annotate(
                str(
                    meta_analysis.iloc[
                        index
                    ]["number"]
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
        "Real Saffron vs Known Artificial Color"
    )

    plt.legend()

    plt.tight_layout()

    pca_plot_path = (
        OUTPUT_DIR
        / "signal_pca.png"
    )

    plt.savefig(
        pca_plot_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # PERMUTATION PLOT
    # --------------------------------------------------------

    print(
        "[8] PERMUTATION PLOT"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.hist(
        null_distances,
        bins=40,
        alpha=0.8,
    )

    plt.axvline(
        observed,
        linewidth=2,
        label="Observed distance",
    )

    plt.xlabel(
        "Centroid distance"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Permutation Test"
    )

    plt.legend()

    plt.tight_layout()

    permutation_plot_path = (
        OUTPUT_DIR
        / "permutation_test.png"
    )

    plt.savefig(
        permutation_plot_path,
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # SAMPLE TABLE
    # --------------------------------------------------------

    sample_table = (
        meta_analysis[
            [
                "number",
                "name",
                "class",
                "subclass",
                "artificial_color",
                "artificial_color_percent",
            ]
        ]
        .copy()
    )

    sample_table[
        "analysis_label"
    ] = labels

    sample_table_path = (
        OUTPUT_DIR
        / "analysis_samples.csv"
    )

    sample_table.to_csv(
        sample_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # DASHBOARD JSON
    # --------------------------------------------------------

    result_status = (
        "signal_detected"
        if p_value < 0.05
        else "no_clear_signal"
    )

    dashboard_payload = {
        "analysis": {
            "name":
                "Artificial Color Signal Analysis",

            "region": {
                "lower_ppm":
                    COLOR_REGION[0],

                "upper_ppm":
                    COLOR_REGION[1],
            },

            "sample_count":
                int(len(labels)),

            "confirmed_real_saffron":
                n_negative,

            "known_artificial_color":
                n_positive,

            "excluded_samples":
                int(len(meta) - len(labels)),
        },

        "metrics": {
            "observed_centroid_distance":
                float(observed_distance),

            "null_mean":
                null_mean,

            "null_std":
                null_std,

            "permutation_z_score":
                float(z_score)
                if not np.isnan(z_score)
                else None,

            "permutation_p_value":
                p_value,

            "permutations":
                N_PERMUTATIONS,
        },

        "decision": {
            "status":
                result_status,

            "alpha":
                0.05,
        },

        "pca": {
            "components":
                int(n_components),

            "pc1_explained_percent":
                float(
                    explained[0] * 100
                ),

            "pc2_explained_percent":
                float(
                    explained[1] * 100
                ),

            "cumulative_explained_percent":
                float(
                    cumulative[-1] * 100
                ),
        },

        "artifacts": {
            "pca_scores":
                str(scores_path),

            "top_discriminating_ppm":
                str(top_features_path),

            "pca_plot":
                str(pca_plot_path),

            "permutation_plot":
                str(permutation_plot_path),

            "analysis_samples":
                str(sample_table_path),
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
            dashboard_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLOR SIGNAL ANALYSIS COMPLETED")
    print("=" * 70)

    print()
    print(
        f"p-value : "
        f"{p_value:.6f}"
    )

    print(
        f"Status  : "
        f"{result_status}"
    )

    print()
    print(
        f"Dashboard JSON : "
        f"{dashboard_path}"
    )


if __name__ == "__main__":
    main()