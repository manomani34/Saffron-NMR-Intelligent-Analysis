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
    "reports/color_loo_validation"
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

    std[std == 0] = 1.0

    return (
        X - mean
    ) / std


def cosine_similarity(a, b):
    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def euclidean_distance(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def robust_z(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    median = np.median(values)

    mad = np.median(
        np.abs(
            values - median
        )
    )

    if mad > 0:
        return (
            values - median
        ) / (
            1.4826 * mad
        )

    std = np.std(values)

    if std > 0:
        return (
            values - median
        ) / std

    return np.zeros_like(values)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR LEAVE-ONE-OUT REFERENCE VALIDATION")
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

    region_mask = (
        (ppm >= COLOR_REGION[0])
        & (ppm <= COLOR_REGION[1])
    )

    X_region = X[
        :,
        region_mask,
    ]

    print(
        f"Region X : "
        f"{X_region.shape}"
    )

    # --------------------------------------------------------
    # SNV
    # --------------------------------------------------------

    X_snv = snv(
        X_region
    )

    # --------------------------------------------------------
    # REFERENCE INDICES
    # --------------------------------------------------------

    real_indices = np.where(
        (
            meta["class"]
            == "real_saffron"
        ).to_numpy()
    )[0]

    color_indices = np.where(
        (
            meta["class"]
            == "pure_artificial_color"
        ).to_numpy()
    )[0]

    print()
    print("[1] REFERENCES")

    print(
        f"Real saffron references : "
        f"{len(real_indices)}"
    )

    print(
        f"Pure color references   : "
        f"{len(color_indices)}"
    )

    if len(real_indices) < 2:
        raise ValueError(
            "At least two real-saffron reference samples are required."
        )

    if len(color_indices) < 2:
        raise ValueError(
            "At least two pure-color reference samples are required."
        )

    # --------------------------------------------------------
    # LOO SCORING
    # --------------------------------------------------------

    print()
    print("[2] LEAVE-ONE-OUT SCORING")

    rows = []

    for i in range(
        len(X_snv)
    ):

        sample = X_snv[i]

        # --------------------------------------------
        # Real reference excluding current sample
        # --------------------------------------------

        real_ref_indices = [
            index
            for index in real_indices
            if index != i
        ]

        if len(real_ref_indices) == 0:
            real_ref = X_snv[
                real_indices
            ].mean(
                axis=0
            )
        else:
            real_ref = X_snv[
                real_ref_indices
            ].mean(
                axis=0
            )

        # --------------------------------------------
        # Color reference excluding current sample
        # --------------------------------------------

        color_ref_indices = [
            index
            for index in color_indices
            if index != i
        ]

        color_ref = X_snv[
            color_ref_indices
        ].mean(
            axis=0
        )

        # --------------------------------------------
        # Distances
        # --------------------------------------------

        distance_real = (
            euclidean_distance(
                sample,
                real_ref,
            )
        )

        distance_color = (
            euclidean_distance(
                sample,
                color_ref,
            )
        )

        cosine_real = (
            cosine_similarity(
                sample,
                real_ref,
            )
        )

        cosine_color = (
            cosine_similarity(
                sample,
                color_ref,
            )
        )

        distance_margin = (
            distance_real
            - distance_color
        )

        cosine_margin = (
            cosine_color
            - cosine_real
        )

        rows.append(
            {
                "number":
                    int(
                        meta.iloc[i]["number"]
                    ),

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
                    cosine_real,

                "cosine_to_color":
                    cosine_color,

                "cosine_margin":
                    cosine_margin,
            }
        )

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # COMBINED EVIDENCE
    # --------------------------------------------------------

    distance_z = robust_z(
        results[
            "distance_margin"
        ].to_numpy()
    )

    cosine_z = robust_z(
        results[
            "cosine_margin"
        ].to_numpy()
    )

    results[
        "loo_color_score"
    ] = (
        distance_z
        + cosine_z
    ) / 2.0

    # --------------------------------------------------------
    # VALIDATION TARGET
    # --------------------------------------------------------

    results[
        "target"
    ] = np.nan

    results.loc[
        results["class"]
        == "real_saffron",
        "target",
    ] = 0

    results.loc[
        results[
            "known_artificial_color"
        ],
        "target",
    ] = 1

    results = results.dropna(
        subset=["target"]
    ).reset_index(
        drop=True
    )

    results[
        "target"
    ] = results[
        "target"
    ].astype(int)

    print()
    print("[3] VALIDATION COHORT")

    print(
        f"Samples  : "
        f"{len(results)}"
    )

    print(
        f"Negative : "
        f"{np.sum(results['target'] == 0)}"
    )

    print(
        f"Positive : "
        f"{np.sum(results['target'] == 1)}"
    )

    # --------------------------------------------------------
    # THRESHOLD DIAGNOSTIC
    # --------------------------------------------------------

    print()
    print("[4] THRESHOLD DIAGNOSTIC")

    y = results[
        "target"
    ].to_numpy()

    scores = results[
        "loo_color_score"
    ].to_numpy()

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        401,
    )

    threshold_rows = []

    for threshold in thresholds:

        predicted = (
            scores >= threshold
        ).astype(int)

        tp = int(
            np.sum(
                (y == 1)
                & (predicted == 1)
            )
        )

        tn = int(
            np.sum(
                (y == 0)
                & (predicted == 0)
            )
        )

        fp = int(
            np.sum(
                (y == 0)
                & (predicted == 1)
            )
        )

        fn = int(
            np.sum(
                (y == 1)
                & (predicted == 0)
            )
        )

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        balanced_accuracy = (
            sensitivity
            + specificity
        ) / 2.0

        threshold_rows.append(
            {
                "threshold":
                    float(threshold),

                "sensitivity":
                    float(sensitivity),

                "specificity":
                    float(specificity),

                "balanced_accuracy":
                    float(balanced_accuracy),

                "tp":
                    tp,

                "tn":
                    tn,

                "fp":
                    fp,

                "fn":
                    fn,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    best = (
        threshold_df
        .sort_values(
            [
                "balanced_accuracy",
                "specificity",
                "sensitivity",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Best threshold     : "
        f"{best['threshold']:.6f}"
    )

    print(
        f"Balanced accuracy  : "
        f"{best['balanced_accuracy']:.4f}"
    )

    print(
        f"Sensitivity        : "
        f"{best['sensitivity']:.4f}"
    )

    print(
        f"Specificity        : "
        f"{best['specificity']:.4f}"
    )

    # --------------------------------------------------------
    # SAMPLE DECISIONS
    # --------------------------------------------------------

    best_threshold = float(
        best["threshold"]
    )

    results[
        "predicted_artificial_color"
    ] = (
        results[
            "loo_color_score"
        ]
        >= best_threshold
    )

    results[
        "correct"
    ] = (
        results[
            "predicted_artificial_color"
        ].astype(int)
        == results["target"]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    scores_path = (
        OUTPUT_DIR
        / "loo_scores.csv"
    )

    results.to_csv(
        scores_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_path = (
        OUTPUT_DIR
        / "threshold_scan.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    # --------------------------------------------------------
    # DASHBOARD JSON
    # --------------------------------------------------------

    sample_results = []

    for _, row in results.iterrows():

        sample_results.append(
            {
                "number":
                    int(row["number"]),

                "name":
                    str(row["name"]),

                "class":
                    str(row["class"]),

                "score":
                    float(
                        row[
                            "loo_color_score"
                        ]
                    ),

                "predictedArtificialColor":
                    bool(
                        row[
                            "predicted_artificial_color"
                        ]
                    ),

                "correct":
                    bool(
                        row["correct"]
                    ),
            }
        )

    dashboard = {
        "analysis": {
            "name":
                "Leave-One-Out Color Validation",

            "region": {
                "lowerPpm":
                    COLOR_REGION[0],

                "upperPpm":
                    COLOR_REGION[1],
            },

            "method":
                "Leave-one-out reference scoring",

            "exploratory":
                True,

            "independentlyValidated":
                False,
        },

        "cohort": {
            "total":
                int(len(results)),

            "negative":
                int(np.sum(y == 0)),

            "positive":
                int(np.sum(y == 1)),
        },

        "metrics": {
            "bestThreshold":
                best_threshold,

            "balancedAccuracy":
                float(
                    best[
                        "balanced_accuracy"
                    ]
                ),

            "sensitivity":
                float(
                    best["sensitivity"]
                ),

            "specificity":
                float(
                    best["specificity"]
                ),
        },

        "samples":
            sample_results,

        "limitations": [
            "Only two confirmed real-saffron reference samples are available.",
            "Threshold selection is still performed on the pilot dataset.",
            "This is not independent external validation.",
            "The score is not a calibrated probability.",
        ],

        "artifacts": {
            "scores":
                str(scores_path),

            "thresholdScan":
                str(threshold_path),
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

    print()
    print("=" * 70)
    print("LEAVE-ONE-OUT VALIDATION COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Scores CSV     : "
        f"{scores_path}"
    )

    print(
        f"Threshold CSV  : "
        f"{threshold_path}"
    )

    print(
        f"Dashboard JSON : "
        f"{dashboard_path}"
    )


if __name__ == "__main__":
    main()