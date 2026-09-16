from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SCORES_PATH = Path(
    "reports/color_detection_score/color_detection_scores.csv"
)

META_PATH = Path(
    "data/processed/color_preprocessed.csv"
)

OUTPUT_DIR = Path(
    "reports/color_threshold_validation"
)

THRESHOLDS = np.arange(
    -1.5,
    1.51,
    0.05,
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    scores = pd.read_csv(
        SCORES_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    return scores, meta


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR SCORE THRESHOLD VALIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores, meta = load_data()

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_meta = meta[
        [
            "number",
            "class",
            "known_artificial_color",
        ]
    ].copy()

    target_meta[
        "target"
    ] = (
        target_meta[
            "known_artificial_color"
        ].astype(bool)
        |
        (
            target_meta["class"]
            == "real_saffron"
        )
        .map(lambda value: not value)
        .astype(bool)
    )

    # --------------------------------------------------------
    # USE EXPLICIT LABEL
    # --------------------------------------------------------

    target_meta["target"] = np.nan

    target_meta.loc[
        target_meta["class"]
        == "real_saffron",
        "target",
    ] = 0

    target_meta.loc[
        target_meta[
            "known_artificial_color"
        ].astype(bool),
        "target",
    ] = 1

    target_meta = target_meta.dropna(
        subset=["target"]
    )

    target_meta[
        "target"
    ] = target_meta[
        "target"
    ].astype(int)

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = scores.merge(
        target_meta[
            [
                "number",
                "target",
            ]
        ],
        on="number",
        how="inner",
        validate="one_to_one",
    )

    y = df[
        "target"
    ].to_numpy()

    score = df[
        "color_evidence_score"
    ].to_numpy()

    print()
    print("[1] VALIDATION COHORT")

    print(
        f"Samples : {len(df)}"
    )

    print(
        f"Negative : "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Positive : "
        f"{np.sum(y == 1)}"
    )

    # --------------------------------------------------------
    # THRESHOLD SCAN
    # --------------------------------------------------------

    print()
    print("[2] THRESHOLD SCAN")

    rows = []

    for threshold in THRESHOLDS:

        predicted = (
            score >= threshold
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

        accuracy = (
            (tp + tn)
            / len(y)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        f1 = (
            2 * precision * sensitivity
            / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )

        youden_j = (
            sensitivity
            + specificity
            - 1.0
        )

        rows.append(
            {
                "threshold":
                    float(threshold),

                "accuracy":
                    float(accuracy),

                "balanced_accuracy":
                    float(balanced_accuracy),

                "sensitivity":
                    float(sensitivity),

                "specificity":
                    float(specificity),

                "precision":
                    float(precision),

                "f1":
                    float(f1),

                "youden_j":
                    float(youden_j),

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
        rows
    )

    threshold_df = (
        threshold_df
        .sort_values(
            [
                "balanced_accuracy",
                "youden_j",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # BEST THRESHOLD
    # --------------------------------------------------------

    best = threshold_df.iloc[0]

    best_threshold = float(
        best["threshold"]
    )

    print()
    print("[3] BEST THRESHOLD")

    print(
        f"Threshold          : "
        f"{best_threshold:.2f}"
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

    print(
        f"Precision          : "
        f"{best['precision']:.4f}"
    )

    print(
        f"F1                 : "
        f"{best['f1']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE THRESHOLDS
    # --------------------------------------------------------

    threshold_path = (
        OUTPUT_DIR
        / "threshold_scan.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAMPLE DECISIONS
    # --------------------------------------------------------

    decision_df = df[
        [
            "number",
            "name",
            "class",
            "color_evidence_score",
        ]
    ].copy()

    decision_df[
        "predicted_artificial_color"
    ] = (
        decision_df[
            "color_evidence_score"
        ]
        >= best_threshold
    )

    decision_df[
        "correct"
    ] = (
        decision_df[
            "predicted_artificial_color"
        ]
        == y
    )

    decision_df = decision_df.sort_values(
        "color_evidence_score",
        ascending=False,
    )

    decision_path = (
        OUTPUT_DIR
        / "validated_decisions.csv"
    )

    decision_df.to_csv(
        decision_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # DASHBOARD JSON
    # --------------------------------------------------------

    dashboard = {

        "analysis": {
            "name":
                "Artificial Color Threshold Validation",

            "method":
                "Threshold scan on exploratory color evidence score",

            "exploratory":
                True,

            "validated":
                False,
        },

        "cohort": {
            "samples":
                int(len(df)),

            "negative":
                int(np.sum(y == 0)),

            "positive":
                int(np.sum(y == 1)),
        },

        "bestThreshold": {
            "value":
                best_threshold,

            "accuracy":
                float(best["accuracy"]),

            "balancedAccuracy":
                float(
                    best[
                        "balanced_accuracy"
                    ]
                ),

            "sensitivity":
                float(
                    best[
                        "sensitivity"
                    ]
                ),

            "specificity":
                float(
                    best[
                        "specificity"
                    ]
                ),

            "precision":
                float(
                    best[
                        "precision"
                    ]
                ),

            "f1":
                float(
                    best["f1"]
                ),

            "youdenJ":
                float(
                    best["youden_j"]
                ),
        },

        "limitations": [
            "Threshold selection uses the same small pilot dataset used to construct the exploratory score.",
            "The threshold is not independently validated.",
            "The score is not a calibrated probability.",
            "Only two confirmed real-saffron references are available.",
        ],

        "artifacts": {
            "thresholdScan":
                str(threshold_path),

            "validatedDecisions":
                str(decision_path),
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
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("THRESHOLD VALIDATION COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Threshold CSV : "
        f"{threshold_path}"
    )

    print(
        f"Decision CSV  : "
        f"{decision_path}"
    )

    print(
        f"Dashboard JSON: "
        f"{dashboard_path}"
    )


if __name__ == "__main__":
    main()