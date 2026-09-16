from pathlib import Path

import numpy as np
import pandas as pd


def _safe_float(value):
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return np.nan


def _load_origin_model_status(
    robust_path: str,
):
    file_path = Path(
        robust_path
    )

    if not file_path.exists():
        return (
            "NOT AVAILABLE",
            np.nan,
            np.nan,
            np.nan,
        )

    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
        )

        if df.empty:
            return (
                "NOT AVAILABLE",
                np.nan,
                np.nan,
                np.nan,
            )

        required = {
            "TopK",
            "AccuracyMean",
            "BalancedAccuracyMean",
            "F1MacroMean",
        }

        if not required.issubset(
            df.columns
        ):
            return (
                "NOT AVAILABLE",
                np.nan,
                np.nan,
                np.nan,
            )

        df = df.sort_values(
            by=[
                "BalancedAccuracyMean",
                "F1MacroMean",
                "AccuracyMean",
            ],
            ascending=False,
        )

        best = df.iloc[0]

        balanced_accuracy = _safe_float(
            best[
                "BalancedAccuracyMean"
            ]
        )

        macro_f1 = _safe_float(
            best[
                "F1MacroMean"
            ]
        )

        top_k = _safe_float(
            best["TopK"]
        )

        # The project currently has insufficient
        # evidence for an operationally reliable
        # origin classifier.
        #
        # Above-chance performance is retained as
        # a descriptive candidate signal only.

        status = "NOT RELIABLE"

        return (
            status,
            balanced_accuracy,
            macro_f1,
            top_k,
        )

    except Exception:
        return (
            "NOT AVAILABLE",
            np.nan,
            np.nan,
            np.nan,
        )


def run_decision_engine(
    novelty_path: str = (
        "novelty_detection_results.csv"
    ),
    predictions_path: str = (
        "sample_predictions.csv"
    ),
    robust_path: str = (
        "robust_evaluation_summary.csv"
    ),
    output_path: str = (
        "decision_engine_results.csv"
    ),
) -> pd.DataFrame:

    print(
        "\n" + "=" * 70
    )

    print(
        "DECISION ENGINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------
    # 1. Load novelty results
    # --------------------------------------------------

    print(
        "\n[1] LOADING NOVELTY RESULTS"
    )

    novelty_file = Path(
        novelty_path
    )

    if not novelty_file.exists():
        raise FileNotFoundError(
            f"Novelty results not found: "
            f"{novelty_path}"
        )

    novelty_df = pd.read_csv(
        novelty_file,
        encoding="utf-8-sig",
    )

    if novelty_df.empty:
        raise ValueError(
            "Novelty results are empty."
        )

    required_novelty_columns = {
        "SampleId",
        "MahalanobisSquared",
        "ThresholdSquared",
        "IsNovel",
        "NoveltyStatus",
    }

    missing_novelty = (
        required_novelty_columns
        - set(
            novelty_df.columns
        )
    )

    if missing_novelty:
        raise ValueError(
            "Missing novelty columns: "
            f"{sorted(missing_novelty)}"
        )

    print(
        f"Novelty records : "
        f"{len(novelty_df)}"
    )

    # --------------------------------------------------
    # 2. Load OOF predictions
    # --------------------------------------------------

    print(
        "\n[2] LOADING ORIGIN PREDICTIONS"
    )

    predictions_df = None

    predictions_file = Path(
        predictions_path
    )

    if predictions_file.exists():

        try:
            predictions_df = pd.read_csv(
                predictions_file,
                encoding="utf-8-sig",
            )

            print(
                "Prediction file status : AVAILABLE"
            )

        except Exception as exc:

            print(
                "WARNING: Could not load prediction "
                f"file: {exc}"
            )

            predictions_df = None

    else:

        print(
            "Prediction file status : NOT AVAILABLE"
        )

    # --------------------------------------------------
    # 3. Robust origin model status
    # --------------------------------------------------

    print(
        "\n[3] ROBUST ORIGIN MODEL STATUS"
    )

    (
        origin_status,
        origin_balanced_accuracy,
        origin_macro_f1,
        origin_top_k,
    ) = _load_origin_model_status(
        robust_path
    )

    print(
        f"Origin model status      : "
        f"{origin_status}"
    )

    if np.isfinite(
        origin_balanced_accuracy
    ):

        print(
            f"Robust Balanced Accuracy : "
            f"{origin_balanced_accuracy:.4f}"
        )

    if np.isfinite(
        origin_macro_f1
    ):

        print(
            f"Robust Macro F1          : "
            f"{origin_macro_f1:.4f}"
        )

    if np.isfinite(
        origin_top_k
    ):

        print(
            f"Robust Top-K             : "
            f"{int(origin_top_k)}"
        )

    # --------------------------------------------------
    # 4. Prepare novelty table
    # --------------------------------------------------

    print(
        "\n[4] MERGING DECISION SIGNALS"
    )

    results = novelty_df.copy()

    results["SampleId"] = (
        pd.to_numeric(
            results["SampleId"],
            errors="coerce",
        )
    )

    results = results[
        results["SampleId"].notna()
    ].copy()

    results["SampleId"] = (
        results["SampleId"]
        .astype(int)
    )

    # --------------------------------------------------
    # Merge OOF predictions as diagnostic information
    # --------------------------------------------------

    if predictions_df is not None:

        merge_columns = [
            "SampleId",
            "ActualGroup",
            "PredictedGroup",
        ]

        optional_columns = [
            "PredictionConfidence",
            "Correct",
            "FinalModelPrediction",
            "FinalModelConfidence",
        ]

        for column in optional_columns:

            if column in predictions_df.columns:
                merge_columns.append(
                    column
                )

        if all(
            column in predictions_df.columns
            for column in [
                "SampleId",
                "ActualGroup",
                "PredictedGroup",
            ]
        ):

            origin_merge = (
                predictions_df[
                    merge_columns
                ].copy()
            )

            origin_merge["SampleId"] = (
                pd.to_numeric(
                    origin_merge[
                        "SampleId"
                    ],
                    errors="coerce",
                )
            )

            origin_merge = (
                origin_merge[
                    origin_merge[
                        "SampleId"
                    ].notna()
                ].copy()
            )

            origin_merge["SampleId"] = (
                origin_merge[
                    "SampleId"
                ].astype(int)
            )

            origin_merge = (
                origin_merge
                .drop_duplicates(
                    subset=[
                        "SampleId"
                    ],
                    keep="first",
                )
            )

            results = results.merge(
                origin_merge,
                on="SampleId",
                how="left",
            )

    # Ensure expected columns exist.
    for column in [
        "ActualGroup",
        "PredictedGroup",
        "PredictionConfidence",
        "Correct",
        "FinalModelPrediction",
        "FinalModelConfidence",
    ]:

        if column not in results.columns:

            results[column] = np.nan

    # --------------------------------------------------
    # 5. Origin decision
    # --------------------------------------------------

    print(
        "\n[5] ORIGIN DECISION"
    )

    if (
        origin_status
        == "RELIABLE"
    ):

        results[
            "OriginDecision"
        ] = np.where(
            results[
                "PredictedGroup"
            ].notna(),
            np.where(
                results[
                    "ActualGroup"
                ].astype(str)
                ==
                results[
                    "PredictedGroup"
                ].astype(str),
                "ORIGIN CONSISTENT",
                "ORIGIN INCONSISTENT - "
                "EXPERT REVIEW REQUIRED",
            ),
            "ORIGIN PREDICTION UNAVAILABLE",
        )

    elif (
        origin_status
        == "NOT RELIABLE"
    ):

        results[
            "OriginDecision"
        ] = (
            "ORIGIN MODEL NOT RELIABLE - "
            "MORE REFERENCE DATA REQUIRED"
        )

    else:

        results[
            "OriginDecision"
        ] = (
            "ORIGIN PREDICTION NOT AVAILABLE"
        )

    # --------------------------------------------------
    # 6. Novelty
    # --------------------------------------------------

    print(
        "\n[6] NOVELTY DECISION"
    )

    results["IsNovel"] = (
        results["IsNovel"]
        .astype(bool)
    )

    results[
        "NoveltyDecision"
    ] = np.where(
        results["IsNovel"],
        "NOVEL / OOD CANDIDATE",
        "WITHIN REFERENCE DOMAIN",
    )

    # --------------------------------------------------
    # 7. Authenticity / adulteration
    # --------------------------------------------------

    print(
        "\n[7] AUTHENTICITY / ADULTERATION"
    )

    results[
        "AdulterationStatus"
    ] = (
        "NOT ASSESSED / "
        "REQUIRES EXPERT REVIEW"
    )

    # --------------------------------------------------
    # 8. Final decision
    # --------------------------------------------------

    print(
        "\n[8] FINAL DECISION"
    )

    final_decisions = []

    for _, row in results.iterrows():

        is_novel = bool(
            row["IsNovel"]
        )

        if is_novel:

            final_decisions.append(
                "NOVEL / OOD CANDIDATE - "
                "EXPERT REVIEW REQUIRED"
            )

            continue

        if (
            origin_status
            == "NOT RELIABLE"
        ):

            final_decisions.append(
                "WITHIN REFERENCE DOMAIN - "
                "ORIGIN MODEL NOT RELIABLE"
            )

            continue

        if (
            origin_status
            == "NOT AVAILABLE"
        ):

            final_decisions.append(
                "WITHIN REFERENCE DOMAIN - "
                "ORIGIN NOT AVAILABLE"
            )

            continue

        actual_group = str(
            row["ActualGroup"]
        )

        predicted_group = str(
            row["PredictedGroup"]
        )

        if (
            actual_group
            != predicted_group
        ):

            final_decisions.append(
                "ORIGIN INCONSISTENT - "
                "EXPERT REVIEW REQUIRED"
            )

            continue

        final_decisions.append(
            "WITHIN REFERENCE DOMAIN - "
            "ADULTERATION NOT ASSESSED"
        )

    results[
        "FinalDecision"
    ] = final_decisions

    # --------------------------------------------------
    # 9. Summary
    # --------------------------------------------------

    print(
        "\n[9] DECISION SUMMARY"
    )

    total_samples = len(
        results
    )

    novel_count = int(
        results[
            "IsNovel"
        ].sum()
    )

    non_novel_count = int(
        (
            ~results[
                "IsNovel"
            ]
        ).sum()
    )

    print(
        f"Total samples : "
        f"{total_samples}"
    )

    print(
        f"Novel / OOD   : "
        f"{novel_count}"
    )

    print(
        f"Non-novel     : "
        f"{non_novel_count}"
    )

    print(
        "Adulteration  : "
        "NOT ASSESSED / REQUIRES EXPERT REVIEW"
    )

    # --------------------------------------------------
    # 10. Save
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

    print(
        "\n" + "=" * 70
    )

    print(
        "DECISION ENGINE FINISHED"
    )

    print(
        "=" * 70
    )

    return results