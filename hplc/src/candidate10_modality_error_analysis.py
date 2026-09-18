from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = (
    ROOT
    / "outputs"
)

STABLE_ERRORS_FILE = (
    OUTPUTS_DIR
    / "candidate10_stable_errors.csv"
)

CANDIDATE10_FILE = (
    OUTPUTS_DIR
    / "candidate10_predictions.csv"
)

RESULT_FILE = (
    OUTPUTS_DIR
    / "candidate10_modality_error_analysis.csv"
)

SUMMARY_FILE = (
    OUTPUTS_DIR
    / "candidate10_modality_error_summary.csv"
)


# ============================================================
# TARGET SAMPLES
# ============================================================

DEFAULT_PROBLEM_SAMPLES = [
    1,
    3,
    8,
    11,
    12,
    18,
    20,
    31,
    38,
    42,
]


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(columns, candidates):

    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in normalized:
            return normalized[key]

    for column in columns:

        text = (
            str(column)
            .strip()
            .lower()
        )

        for candidate in candidates:

            candidate_text = (
                str(candidate)
                .strip()
                .lower()
            )

            if candidate_text in text:
                return column

    return None


def detect_prediction_columns(df):

    sample_col = find_column(
        df.columns,
        [
            "SampleId",
            "Sample ID",
            "Sample_ID",
            "sample_id",
            "sample",
            "number",
        ],
    )

    actual_col = find_column(
        df.columns,
        [
            "ActualClass",
            "Actual Class",
            "Actual",
            "TrueClass",
            "True Class",
            "Group",
        ],
    )

    predicted_col = find_column(
        df.columns,
        [
            "PredictedClass",
            "Predicted Class",
            "Prediction",
            "Predicted",
            "PredClass",
        ],
    )

    return (
        sample_col,
        actual_col,
        predicted_col,
    )


# ============================================================
# SCENARIO DETECTION
# ============================================================

def detect_scenario_columns(df):

    candidates = [
        "Scenario",
        "Modality",
        "Wavelength",
        "Preprocessing",
        "Preprocess",
        "Config",
        "Configuration",
        "Model",
        "Setting",
        "Method",
    ]

    result = []

    for candidate in candidates:

        column = find_column(
            df.columns,
            [candidate],
        )

        if (
            column is not None
            and column not in result
        ):
            result.append(column)

    return result


def normalize_scenario_value(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    text_lower = (
        text
        .lower()
        .replace(" ", "")
    )

    if "combined" in text_lower:
        return "Combined"

    # Detect exact wavelength values.
    if re.search(
        r"(^|[^0-9])440([^0-9]|$)",
        text_lower,
    ):
        return "440"

    if re.search(
        r"(^|[^0-9])308([^0-9]|$)",
        text_lower,
    ):
        return "308"

    if re.search(
        r"(^|[^0-9])250([^0-9]|$)",
        text_lower,
    ):
        return "250"

    return text


def build_scenario_series(
    df,
    scenario_columns,
):

    if not scenario_columns:

        return pd.Series(
            [""] * len(df),
            index=df.index,
        )

    values = []

    for _, row in df.iterrows():

        parts = []

        for column in scenario_columns:

            value = row[column]

            if pd.isna(value):
                continue

            text = str(value).strip()

            if text:
                parts.append(text)

        combined = " | ".join(parts)

        values.append(
            normalize_scenario_value(
                combined
            )
        )

    return pd.Series(
        values,
        index=df.index,
    )


# ============================================================
# LOAD PREDICTION FILE
# ============================================================

def load_prediction_file(
    file_path,
):

    try:

        df = pd.read_csv(
            file_path
        )

    except Exception:
        return None

    (
        sample_col,
        actual_col,
        predicted_col,
    ) = detect_prediction_columns(
        df
    )

    if (
        sample_col is None
        or actual_col is None
        or predicted_col is None
    ):
        return None

    scenario_columns = (
        detect_scenario_columns(
            df
        )
    )

    result = pd.DataFrame()

    result["SampleId"] = (
        pd.to_numeric(
            df[sample_col],
            errors="coerce",
        )
    )

    result["ActualClass"] = (
        df[actual_col]
        .astype(str)
        .str.strip()
    )

    result["PredictedClass"] = (
        df[predicted_col]
        .astype(str)
        .str.strip()
    )

    result["Scenario"] = (
        build_scenario_series(
            df,
            scenario_columns,
        )
    )

    result["SourceFile"] = (
        file_path.name
    )

    result = result[
        result["SampleId"].notna()
    ].copy()

    result["SampleId"] = (
        result["SampleId"]
        .astype(int)
    )

    return result


# ============================================================
# DISCOVER PREDICTION FILES
# ============================================================

def discover_prediction_files():

    files = []

    for file_path in sorted(
        OUTPUTS_DIR.glob("*.csv")
    ):

        result = load_prediction_file(
            file_path
        )

        if result is None:
            continue

        if result.empty:
            continue

        files.append(
            (
                file_path,
                result,
            )
        )

    return files


# ============================================================
# CLASSIFY SCENARIO
# ============================================================

def classify_scenario(
    scenario,
):

    text = (
        str(scenario)
        .strip()
        .lower()
        .replace(" ", "")
    )

    if not text:
        return None

    if "combined" in text:
        return "Combined"

    if re.search(
        r"(^|[^0-9])440([^0-9]|$)",
        text,
    ):
        return "440"

    if re.search(
        r"(^|[^0-9])308([^0-9]|$)",
        text,
    ):
        return "308"

    if re.search(
        r"(^|[^0-9])250([^0-9]|$)",
        text,
    ):
        return "250"

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 78
    )
    print(
        "CANDIDATE 10 MODALITY ERROR ANALYSIS"
    )
    print(
        "=" * 78
    )

    if not OUTPUTS_DIR.exists():

        raise FileNotFoundError(
            f"Outputs directory not found:\n"
            f"{OUTPUTS_DIR}"
        )

    # ========================================================
    # TARGET SAMPLE IDS
    # ========================================================

    target_sample_ids = (
        DEFAULT_PROBLEM_SAMPLES.copy()
    )

    if STABLE_ERRORS_FILE.exists():

        try:

            stable = pd.read_csv(
                STABLE_ERRORS_FILE
            )

            if "SampleId" in stable.columns:

                ids = (
                    pd.to_numeric(
                        stable["SampleId"],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                    .tolist()
                )

                if ids:
                    target_sample_ids = ids

        except Exception:
            pass

    print()
    print(
        "Target samples:"
    )

    print(
        target_sample_ids
    )

    # ========================================================
    # DISCOVER FILES
    # ========================================================

    discovered = (
        discover_prediction_files()
    )

    print()
    print(
        "Prediction files detected:"
    )

    for file_path, df in discovered:

        print(
            f"  {file_path.name}"
            f"  rows={len(df)}"
            f"  scenarios="
            f"{sorted(df['Scenario'].dropna().unique().tolist())[:10]}"
        )

    if not discovered:

        raise RuntimeError(
            "No prediction CSV containing "
            "SampleId / ActualClass / PredictedClass "
            "was found in hplc/outputs."
        )

    # ========================================================
    # COMBINE FILES
    # ========================================================

    all_predictions = pd.concat(
        [
            df
            for _, df in discovered
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove duplicate representation of the exact same
    # prediction row coming from multiple output files.
    # --------------------------------------------------------

    all_predictions = (
        all_predictions
        .drop_duplicates(
            subset=[
                "SourceFile",
                "SampleId",
                "ActualClass",
                "PredictedClass",
                "Scenario",
            ]
        )
        .copy()
    )

    # ========================================================
    # NORMALIZE SCENARIO
    # ========================================================

    all_predictions["Modality"] = (
        all_predictions["Scenario"]
        .apply(
            classify_scenario
        )
    )

    # Candidate 10 file is Combined by definition.
    candidate10_mask = (
        all_predictions["SourceFile"]
        == CANDIDATE10_FILE.name
    )

    all_predictions.loc[
        candidate10_mask,
        "Modality",
    ] = "Combined"

    # ========================================================
    # FOCUS ON FOUR MODALITIES
    # ========================================================

    modalities = [
        "250",
        "308",
        "440",
        "Combined",
    ]

    focused = all_predictions[
        all_predictions["Modality"].isin(
            modalities
        )
    ].copy()

    focused = focused[
        focused["SampleId"].isin(
            target_sample_ids
        )
    ].copy()

    # ========================================================
    # PER SAMPLE + MODALITY
    # ========================================================

    rows = []

    for (
        (sample_id, modality),
        group,
    ) in focused.groupby(
        [
            "SampleId",
            "Modality",
        ]
    ):

        actual_counts = (
            group["ActualClass"]
            .value_counts()
        )

        actual = (
            actual_counts
            .index[0]
        )

        prediction_counts = (
            group["PredictedClass"]
            .value_counts()
        )

        dominant_prediction = (
            prediction_counts
            .index[0]
        )

        total = len(group)

        stability = (
            prediction_counts.iloc[0]
            / total
        )

        correct_rate = (
            (
                group["PredictedClass"]
                == actual
            )
            .mean()
        )

        rows.append(
            {
                "SampleId":
                    int(sample_id),

                "ActualClass":
                    actual,

                "Modality":
                    modality,

                "DominantPrediction":
                    dominant_prediction,

                "Stability":
                    float(
                        stability
                    ),

                "CorrectRate":
                    float(
                        correct_rate
                    ),

                "PredictionCount":
                    total,

                "IsCorrect":
                    (
                        dominant_prediction
                        == actual
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        print()
        print(
            "No predictions for the requested "
            "modalities were found."
        )

        print()
        print(
            "Detected scenarios:"
        )

        print(
            sorted(
                all_predictions[
                    "Scenario"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )

        return

    # ========================================================
    # PIVOT
    # ========================================================

    pivot = result.pivot_table(
        index=[
            "SampleId",
            "ActualClass",
        ],
        columns="Modality",
        values=[
            "DominantPrediction",
            "Stability",
            "CorrectRate",
            "IsCorrect",
        ],
        aggfunc="first",
    )

    pivot = pivot.reset_index()

    # ========================================================
    # SUMMARY BY MODALITY
    # ========================================================

    summary_rows = []

    for modality in modalities:

        modality_df = result[
            result["Modality"]
            == modality
        ].copy()

        if modality_df.empty:
            continue

        correct = int(
            modality_df[
                "IsCorrect"
            ].sum()
        )

        total = len(
            modality_df
        )

        summary_rows.append(
            {
                "Modality":
                    modality,

                "ProblemSamplesFound":
                    total,

                "CorrectDominantPrediction":
                    correct,

                "WrongDominantPrediction":
                    total - correct,

                "ProblemSampleAccuracy":
                    correct / total
                    if total
                    else np.nan,

                "MeanStability":
                    float(
                        modality_df[
                            "Stability"
                        ].mean()
                    ),

                "MeanCorrectRate":
                    float(
                        modality_df[
                            "CorrectRate"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # ERROR RESPONSIBILITY
    # ========================================================

    responsibility_rows = []

    for _, row in result.iterrows():

        responsibility_rows.append(
            {
                "SampleId":
                    row["SampleId"],

                "ActualClass":
                    row["ActualClass"],

                "Modality":
                    row["Modality"],

                "DominantPrediction":
                    row[
                        "DominantPrediction"
                    ],

                "Stability":
                    row["Stability"],

                "CorrectRate":
                    row["CorrectRate"],

                "Status":
                    (
                        "CORRECT"
                        if row["IsCorrect"]
                        else "WRONG"
                    ),
            }
        )

    responsibility = pd.DataFrame(
        responsibility_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    result = result.sort_values(
        [
            "SampleId",
            "Modality",
        ]
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "PER SAMPLE / MODALITY"
    )
    print(
        "=" * 78
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "=" * 78
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 78
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAMPLE MATRIX
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "SAMPLE × MODALITY DECISION"
    )
    print(
        "=" * 78
    )

    for sample_id in target_sample_ids:

        sample_df = result[
            result["SampleId"]
            == sample_id
        ].copy()

        if sample_df.empty:
            continue

        actual = (
            sample_df[
                "ActualClass"
            ].iloc[0]
        )

        print()
        print(
            f"Sample {sample_id} "
            f"(Actual={actual})"
        )

        for modality in modalities:

            row = sample_df[
                sample_df["Modality"]
                == modality
            ]

            if row.empty:

                print(
                    f"  {modality:8s}: MISSING"
                )

                continue

            r = row.iloc[0]

            status = (
                "OK"
                if r["IsCorrect"]
                else "WRONG"
            )

            print(
                f"  {modality:8s}: "
                f"{r['DominantPrediction']} "
                f"(stability="
                f"{r['Stability']:.2f}, "
                f"correct-rate="
                f"{r['CorrectRate']:.2f}) "
                f"[{status}]"
            )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "OUTPUTS"
    )
    print(
        "=" * 78
    )

    print(
        f"Detailed : {RESULT_FILE}"
    )

    print(
        f"Summary  : {SUMMARY_FILE}"
    )

    print()


if __name__ == "__main__":
    main()