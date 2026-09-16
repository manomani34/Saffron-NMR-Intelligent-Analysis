from pathlib import Path
import json

import numpy as np
import pandas as pd


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
    "reports/color_detection_score"
)

COLOR_REGION = (
    5.0,
    9.0,
)


# ============================================================
# HELPERS
# ============================================================

def snv(X):
    mean = X.mean(
        axis=1,
        keepdims=True,
    )

    std = X.std(
        axis=1,
        keepdims=True,
    )

    std[
        std == 0
    ] = 1.0

    return (
        X - mean
    ) / std


def cosine_similarity(
    vector_a,
    vector_b,
):
    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        / denominator
    )


def euclidean_distance(
    vector_a,
    vector_b,
):
    return float(
        np.linalg.norm(
            vector_a - vector_b
        )
    )


def robust_scale(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    median = np.median(
        values
    )

    mad = np.median(
        np.abs(
            values - median
        )
    )

    if mad == 0:
        std = np.std(
            values
        )

        if std == 0:
            return np.zeros_like(
                values
            )

        return (
            values - median
        ) / std

    return (
        values - median
    ) / (
        1.4826 * mad
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR DETECTION SCORE")
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
        f"Color region X : "
        f"{X_region.shape}"
    )

    # --------------------------------------------------------
    # SNV
    # --------------------------------------------------------

    X_snv = snv(
        X_region
    )

    # --------------------------------------------------------
    # REFERENCE SPECTRA
    # --------------------------------------------------------

    real_mask = (
        meta["class"]
        == "real_saffron"
    ).to_numpy()

    color_mask = (
        meta["class"]
        == "pure_artificial_color"
    ).to_numpy()

    if real_mask.sum() < 2:
        raise ValueError(
            "At least two real saffron references are required."
        )

    if color_mask.sum() < 2:
        raise ValueError(
            "At least two pure artificial-color references are required."
        )

    real_reference = X_snv[
        real_mask
    ].mean(
        axis=0
    )

    color_reference = X_snv[
        color_mask
    ].mean(
        axis=0
    )

    # --------------------------------------------------------
    # SAMPLE SCORES
    # --------------------------------------------------------

    rows = []

    for i in range(
        len(X_snv)
    ):

        sample = X_snv[i]

        distance_real = (
            euclidean_distance(
                sample,
                real_reference,
            )
        )

        distance_color = (
            euclidean_distance(
                sample,
                color_reference,
            )
        )

        similarity_real = (
            cosine_similarity(
                sample,
                real_reference,
            )
        )

        similarity_color = (
            cosine_similarity(
                sample,
                color_reference,
            )
        )

        distance_margin = (
            distance_real
            - distance_color
        )

        similarity_margin = (
            similarity_color
            - similarity_real
        )

        rows.append(
            {
                "number":
                    meta.iloc[i]["number"],

                "name":
                    meta.iloc[i]["name"],

                "class":
                    meta.iloc[i]["class"],

                "known_artificial_color":
                    bool(
                        meta.iloc[i][
                            "known_artificial_color"
                        ]
                    ),

                "distance_to_real":
                    distance_real,

                "distance_to_color":
                    distance_color,

                "distance_margin":
                    distance_margin,

                "cosine_to_real":
                    similarity_real,

                "cosine_to_color":
                    similarity_color,

                "cosine_margin":
                    similarity_margin,
            }
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # NORMALIZED COLOR EVIDENCE
    # --------------------------------------------------------

    distance_margin = (
        results[
            "distance_margin"
        ].to_numpy()
    )

    similarity_margin = (
        results[
            "cosine_margin"
        ].to_numpy()
    )

    distance_z = robust_scale(
        distance_margin
    )

    similarity_z = robust_scale(
        similarity_margin
    )

    evidence = (
        distance_z
        + similarity_z
    ) / 2.0

    results[
        "color_evidence_score"
    ] = evidence

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    results[
        "color_evidence_rank"
    ] = (
        results[
            "color_evidence_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # QUALITATIVE STATUS
    # --------------------------------------------------------

    def classify_score(score):

        if score >= 1.5:
            return "strong_evidence"

        if score >= 0.5:
            return "moderate_evidence"

        if score <= -1.5:
            return "strong_real_saffron_similarity"

        if score <= -0.5:
            return "moderate_real_saffron_similarity"

        return "ambiguous"

    results[
        "evidence_status"
    ] = (
        results[
            "color_evidence_score"
        ]
        .map(
            classify_score
        )
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results = (
        results
        .sort_values(
            "color_evidence_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # SAVE TABLE
    # --------------------------------------------------------

    result_path = (
        OUTPUT_DIR
        / "color_detection_scores.csv"
    )

    results.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # TOP POSITIVE / NEGATIVE
    # --------------------------------------------------------

    top_positive = (
        results
        .sort_values(
            "color_evidence_score",
            ascending=False,
        )
        .head(10)
    )

    top_negative = (
        results
        .sort_values(
            "color_evidence_score",
            ascending=True,
        )
        .head(10)
    )

    print()
    print("[1] TOP COLOR EVIDENCE")

    for _, row in top_positive.iterrows():

        print(
            f"  #{int(row['number']):>2} "
            f"{row['name']:<40} "
            f"score="
            f"{row['color_evidence_score']:.4f} "
            f"{row['evidence_status']}"
        )

    print()
    print("[2] TOP REAL-SAFFRON SIMILARITY")

    for _, row in top_negative.iterrows():

        print(
            f"  #{int(row['number']):>2} "
            f"{row['name']:<40} "
            f"score="
            f"{row['color_evidence_score']:.4f} "
            f"{row['evidence_status']}"
        )

    # --------------------------------------------------------
    # KNOWN LABEL PERFORMANCE
    # --------------------------------------------------------

    known_mask = (
        meta[
            "known_artificial_color"
        ]
        .astype(bool)
        .to_numpy()
        |
        (
            meta["class"]
            == "real_saffron"
        ).to_numpy()
    )

    known_results = results[
        results["number"].isin(
            meta.loc[
                known_mask,
                "number",
            ]
        )
    ].copy()

    known_positive = known_results[
        known_results[
            "known_artificial_color"
        ]
        == True
    ]

    known_negative = known_results[
        known_results[
            "class"
        ]
        == "real_saffron"
    ]

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    dashboard_rows = []

    for _, row in results.iterrows():

        dashboard_rows.append(
            {
                "number":
                    int(row["number"]),

                "name":
                    str(row["name"]),

                "class":
                    str(row["class"]),

                "colorEvidenceScore":
                    float(
                        row[
                            "color_evidence_score"
                        ]
                    ),

                "colorEvidenceRank":
                    int(
                        row[
                            "color_evidence_rank"
                        ]
                    ),

                "status":
                    str(
                        row[
                            "evidence_status"
                        ]
                    ),

                "distanceToReal":
                    float(
                        row[
                            "distance_to_real"
                        ]
                    ),

                "distanceToColor":
                    float(
                        row[
                            "distance_to_color"
                        ]
                    ),

                "cosineToReal":
                    float(
                        row[
                            "cosine_to_real"
                        ]
                    ),

                "cosineToColor":
                    float(
                        row[
                            "cosine_to_color"
                        ]
                    ),
            }
        )

    dashboard = {

        "analysis": {
            "name":
                "Artificial Color Detection Score",

            "region": {
                "lowerPpm":
                    COLOR_REGION[0],

                "upperPpm":
                    COLOR_REGION[1],
            },

            "method":
                "SNV + real-saffron/color reference similarity",

            "exploratory":
                True,

            "validated":
                False,
        },

        "references": {
            "realSaffronCount":
                int(
                    real_mask.sum()
                ),

            "pureArtificialColorCount":
                int(
                    color_mask.sum()
                ),
        },

        "samples":
            dashboard_rows,

        "interpretation": {
            "strongEvidenceThreshold":
                1.5,

            "moderateEvidenceThreshold":
                0.5,

            "strongRealSimilarityThreshold":
                -1.5,

            "moderateRealSimilarityThreshold":
                -0.5,
        },

        "limitations": [
            "Only two confirmed real-saffron references are available.",
            "Scores are exploratory and are not calibrated probabilities.",
            "The score should not be interpreted as an official adulteration percentage.",
        ],

        "artifacts": {
            "scoresCsv":
                str(result_path),
        },

        "dashboardReady":
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
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLOR DETECTION SCORE COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Scores CSV    : {result_path}"
    )

    print(
        f"Dashboard JSON: {dashboard_path}"
    )


if __name__ == "__main__":
    main()