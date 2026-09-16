from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SCORE_PATH = Path(
    "reports/color_detection_score/color_detection_scores.csv"
)

LOO_PATH = Path(
    "reports/color_loo_validation/loo_scores.csv"
)

THRESHOLD_PATH = Path(
    "reports/color_loo_validation/threshold_scan.csv"
)

META_PATH = Path(
    "data/processed/color_preprocessed.csv"
)

OUTPUT_DIR = Path(
    "reports/color_dashboard"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "color_dashboard_result.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "color_dashboard_samples.csv"
)


# ============================================================
# DECISION
# ============================================================

def determine_decision(
    row,
    threshold,
):
    sample_class = str(
        row["class"]
    )

    score = float(
        row["color_evidence_score"]
    )

    known_color = bool(
        row["known_artificial_color"]
    )

    unknown_adulteration = bool(
        row["unknown_adulteration"]
    )

    saffron = bool(
        row["saffron"]
    )

    adulterated = bool(
        row["adulterated"]
    )

    # --------------------------------------------------------
    # Unknown adulteration
    # --------------------------------------------------------

    if unknown_adulteration:

        return (
            "unknown_adulteration",
            "suspicious",
            "Sample is known to contain unknown adulteration.",
        )

    # --------------------------------------------------------
    # Real saffron
    # --------------------------------------------------------

    if sample_class == "real_saffron":

        return (
            "real_saffron",
            "reference",
            "Confirmed real saffron reference sample.",
        )

    # --------------------------------------------------------
    # Known artificial color
    # --------------------------------------------------------

    if known_color:

        if score >= threshold:

            return (
                "artificial_color_detected",
                "positive",
                "Artificial-color evidence is above the exploratory threshold.",
            )

        return (
            "artificial_color_not_detected",
            "warning",
            "Known artificial-color sample is below the exploratory threshold.",
        )

    # --------------------------------------------------------
    # Adulterated saffron
    # --------------------------------------------------------

    if adulterated and saffron:

        if score >= threshold:

            return (
                "suspicious_artificial_color",
                "suspicious",
                "Adulterated saffron shows artificial-color evidence.",
            )

        return (
            "suspicious_other_adulteration",
            "suspicious",
            "Adulteration is known, but artificial-color evidence is below the exploratory threshold.",
        )

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    if score >= threshold:

        return (
            "suspicious_artificial_color",
            "suspicious",
            "Exploratory artificial-color evidence is above the threshold.",
        )

    return (
        "indeterminate",
        "indeterminate",
        "Current pilot data do not support a definitive conclusion.",
    )


# ============================================================
# CONFIDENCE
# ============================================================

def determine_confidence(
    score,
    threshold,
):
    distance = abs(
        score - threshold
    )

    if distance >= 2.0:
        return "high"

    if distance >= 0.75:
        return "medium"

    return "low"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR DASHBOARD RESULT BUILDER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    required_files = [
        SCORE_PATH,
        LOO_PATH,
        THRESHOLD_PATH,
        META_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    score_df = pd.read_csv(
        SCORE_PATH
    )

    loo_df = pd.read_csv(
        LOO_PATH
    )

    threshold_df = pd.read_csv(
        THRESHOLD_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    print()
    print(
        f"Score rows : {len(score_df)}"
    )

    print(
        f"LOO rows   : {len(loo_df)}"
    )

    print(
        f"Metadata   : {len(meta)}"
    )

    # --------------------------------------------------------
    # VALIDATE SCORE COLUMN
    # --------------------------------------------------------

    required_score_columns = [
        "number",
        "color_evidence_score",
        "distance_to_real",
        "distance_to_color",
        "distance_margin",
        "cosine_to_real",
        "cosine_to_color",
        "cosine_margin",
    ]

    missing_score_columns = [
        column
        for column in required_score_columns
        if column not in score_df.columns
    ]

    if missing_score_columns:

        raise ValueError(
            "Missing score columns: "
            f"{missing_score_columns}"
        )

    # --------------------------------------------------------
    # VALIDATE LOO COLUMN
    # --------------------------------------------------------

    required_loo_columns = [
        "number",
        "loo_color_score",
    ]

    missing_loo_columns = [
        column
        for column in required_loo_columns
        if column not in loo_df.columns
    ]

    if missing_loo_columns:

        raise ValueError(
            "Missing LOO columns: "
            f"{missing_loo_columns}"
        )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    required_threshold_columns = [
        "threshold",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]

    missing_threshold_columns = [
        column
        for column in required_threshold_columns
        if column not in threshold_df.columns
    ]

    if missing_threshold_columns:

        raise ValueError(
            "Missing threshold columns: "
            f"{missing_threshold_columns}"
        )

    best_threshold_row = (
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

    threshold = float(
        best_threshold_row[
            "threshold"
        ]
    )

    print()
    print(
        f"Exploratory threshold : "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # VALIDATE METADATA
    # --------------------------------------------------------

    required_meta_columns = [
        "number",
        "name",
        "class",
        "subclass",
        "known_artificial_color",
        "unknown_adulteration",
        "saffron",
        "adulterated",
        "artificial_color_percent",
    ]

    missing_meta_columns = [
        column
        for column in required_meta_columns
        if column not in meta.columns
    ]

    if missing_meta_columns:

        raise ValueError(
            "Missing metadata columns: "
            f"{missing_meta_columns}"
        )

    # --------------------------------------------------------
    # BASE DATA
    # --------------------------------------------------------

    base = meta[
        required_meta_columns
    ].copy()

    score_selected = score_df[
        required_score_columns
    ].copy()

    loo_selected = loo_df[
        required_loo_columns
    ].copy()

    # --------------------------------------------------------
    # CHECK DUPLICATE NUMBERS
    # --------------------------------------------------------

    if base["number"].duplicated().any():

        raise ValueError(
            "Duplicate sample numbers found in metadata."
        )

    if score_selected["number"].duplicated().any():

        raise ValueError(
            "Duplicate sample numbers found in score file."
        )

    if loo_selected["number"].duplicated().any():

        raise ValueError(
            "Duplicate sample numbers found in LOO file."
        )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    result = base.merge(
        score_selected,
        on="number",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        loo_selected,
        on="number",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # CHECK SCORES
    # --------------------------------------------------------

    if result[
        "color_evidence_score"
    ].isna().any():

        missing_numbers = (
            result.loc[
                result[
                    "color_evidence_score"
                ].isna(),
                "number",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing color scores for samples: "
            f"{missing_numbers}"
        )

    # --------------------------------------------------------
    # DECISIONS
    # --------------------------------------------------------

    decisions = []

    for _, row in result.iterrows():

        decision, status, reason = (
            determine_decision(
                row,
                threshold,
            )
        )

        confidence = (
            determine_confidence(
                float(
                    row[
                        "color_evidence_score"
                    ]
                ),
                threshold,
            )
        )

        decisions.append(
            {
                "decision":
                    decision,

                "status":
                    status,

                "reason":
                    reason,

                "confidence":
                    confidence,
            }
        )

    decisions_df = pd.DataFrame(
        decisions
    )

    result = pd.concat(
        [
            result.reset_index(
                drop=True
            ),
            decisions_df,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # DASHBOARD FIELDS
    # --------------------------------------------------------

    result["score"] = (
        result[
            "color_evidence_score"
        ]
    )

    result["threshold"] = threshold

    result["above_threshold"] = (
        result["score"]
        >= threshold
    )

    result["display_score"] = (
        result["score"]
        .round(4)
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    decision_counts = (
        result[
            "decision"
        ]
        .value_counts()
        .to_dict()
    )

    status_counts = (
        result[
            "status"
        ]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # SAMPLE JSON
    # --------------------------------------------------------

    samples = []

    for _, row in result.iterrows():

        artificial_percent = (
            None
            if pd.isna(
                row[
                    "artificial_color_percent"
                ]
            )
            else float(
                row[
                    "artificial_color_percent"
                ]
            )
        )

        samples.append(
            {
                "number":
                    int(
                        row["number"]
                    ),

                "name":
                    str(
                        row["name"]
                    ),

                "class":
                    str(
                        row["class"]
                    ),

                "subclass":
                    str(
                        row["subclass"]
                    ),

                "decision":
                    str(
                        row["decision"]
                    ),

                "status":
                    str(
                        row["status"]
                    ),

                "reason":
                    str(
                        row["reason"]
                    ),

                "confidence":
                    str(
                        row["confidence"]
                    ),

                "score":
                    float(
                        row["score"]
                    ),

                "threshold":
                    threshold,

                "aboveThreshold":
                    bool(
                        row[
                            "above_threshold"
                        ]
                    ),

                "knownArtificialColor":
                    bool(
                        row[
                            "known_artificial_color"
                        ]
                    ),

                "unknownAdulteration":
                    bool(
                        row[
                            "unknown_adulteration"
                        ]
                    ),

                "saffron":
                    bool(
                        row["saffron"]
                    ),

                "adulterated":
                    bool(
                        row["adulterated"]
                    ),

                "artificialColorPercent":
                    artificial_percent,
            }
        )

    # --------------------------------------------------------
    # DASHBOARD JSON
    # --------------------------------------------------------

    dashboard = {

        "version":
            "1.0",

        "dataset":
            "color",

        "analysis": {

            "name":
                "Artificial Color Detection",

            "region": {

                "lowerPpm":
                    5.0,

                "upperPpm":
                    9.0,
            },

            "method":
                "SNV + reference scoring + leave-one-out exploratory validation",

            "exploratory":
                True,

            "productionReady":
                False,
        },

        "threshold": {

            "value":
                threshold,

            "balancedAccuracy":
                float(
                    best_threshold_row[
                        "balanced_accuracy"
                    ]
                ),

            "sensitivity":
                float(
                    best_threshold_row[
                        "sensitivity"
                    ]
                ),

            "specificity":
                float(
                    best_threshold_row[
                        "specificity"
                    ]
                ),

            "type":
                "exploratory",
        },

        "summary": {

            "totalSamples":
                int(
                    len(result)
                ),

            "decisionCounts":
                {
                    str(key): int(value)
                    for key, value
                    in decision_counts.items()
                },

            "statusCounts":
                {
                    str(key): int(value)
                    for key, value
                    in status_counts.items()
                },
        },

        "samples":
            samples,

        "ui": {

            "showScore":
                True,

            "showConfidence":
                True,

            "showReason":
                True,

            "showThreshold":
                True,

            "showArtificialColorPercent":
                True,

            "showWarnings":
                True,
        },

        "limitations": [

            "Pilot dataset contains only two confirmed real-saffron samples.",

            "Threshold is exploratory and not independently validated.",

            "Score is not a calibrated probability.",

            "Unknown adulteration requires whole-spectrum analysis.",

            "A negative artificial-color result does not prove authenticity.",
        ],

        "artifacts": {

            "sampleResults":
                str(
                    OUTPUT_CSV
                ),

            "sourceScores":
                str(
                    SCORE_PATH
                ),

            "looValidation":
                str(
                    LOO_PATH
                ),

            "thresholdScan":
                str(
                    THRESHOLD_PATH
                ),
        },

        "readyForMainDashboard":
            True,
    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
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
    # PRINT
    # --------------------------------------------------------

    print()
    print("[1] DECISION DISTRIBUTION")

    for key, value in decision_counts.items():

        print(
            f"  {key:<40} "
            f"{value:>3}"
        )

    print()
    print("[2] STATUS DISTRIBUTION")

    for key, value in status_counts.items():

        print(
            f"  {key:<20} "
            f"{value:>3}"
        )

    print()
    print("=" * 70)
    print(
        "COLOR DASHBOARD RESULT COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        f"CSV  : {OUTPUT_CSV}"
    )

    print(
        f"JSON : {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()