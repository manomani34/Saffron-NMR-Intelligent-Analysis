from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXCEL_FILE = (
    ROOT
    / "data"
    / "raw"
    / "hplc.xlsx"
)

MAPPING_FILE = (
    ROOT
    / "data"
    / "mapping"
    / "sample_mapping.csv"
)

PREDICTIONS_FILE = (
    ROOT
    / "outputs"
    / "candidate10_predictions.csv"
)

STABLE_ERRORS_FILE = (
    ROOT
    / "outputs"
    / "candidate10_stable_errors.csv"
)

PROFILE_FILE = (
    ROOT
    / "outputs"
    / "candidate10_problem_profiles.csv"
)

SIMILARITY_FILE = (
    ROOT
    / "outputs"
    / "candidate10_problem_similarity.csv"
)


# ============================================================
# SHEETS
# ============================================================

WAVELENGTH_SHEETS = {
    "440": "Crocin 440",
    "250": "picrocrocin 250",
    "308": "safranal 308",
}


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in normalized:
            return normalized[key]

    for c in df.columns:

        text = (
            str(c)
            .strip()
            .lower()
        )

        for candidate in candidates:

            if (
                candidate
                .strip()
                .lower()
                in text
            ):
                return c

    return None


def extract_sample_id(value):

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "",
    }:
        return None

    match = re.search(
        r"\d+",
        text,
    )

    if not match:
        return None

    return int(match.group())


def snv(signal):

    signal = np.asarray(
        signal,
        dtype=float,
    )

    mean = np.mean(signal)
    std = np.std(signal)

    if std <= 1e-12:
        return signal - mean

    return (
        signal - mean
    ) / std


def interpolate_to_reference(
    time,
    signal,
    reference_time,
):

    return np.interp(
        reference_time,
        time,
        signal,
    )


def pearson_similarity(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 3:
        return np.nan

    x = x[valid]
    y = y[valid]

    sx = np.std(x)
    sy = np.std(y)

    if (
        sx <= 1e-12
        or sy <= 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


# ============================================================
# LOAD ONE HPLC SHEET
# ============================================================

def load_wavelength(sheet_name):

    raw = pd.read_excel(
        EXCEL_FILE,
        sheet_name=sheet_name,
        header=None,
    )

    # --------------------------------------------------------
    # Find the real header row.
    #
    # Example:
    #
    # Unnamed Unnamed number Group 0.016667 0.033333 ...
    #
    # or:
    #
    # Unnamed Unnamed Unnamed Unnamed ...
    # ...      ...    Number  Group    0.016667 ...
    # --------------------------------------------------------

    header_row = None

    for i in range(
        min(8, len(raw))
    ):

        values = [
            str(v)
            .strip()
            .lower()
            for v
            in raw.iloc[i].tolist()
            if pd.notna(v)
        ]

        has_number = any(
            v == "number"
            for v in values
        )

        has_group = any(
            v == "group"
            for v in values
        )

        if (
            has_number
            and has_group
        ):
            header_row = i
            break

    if header_row is None:

        raise ValueError(
            "Could not detect the real "
            f"header row in sheet: {sheet_name}"
        )

    # --------------------------------------------------------
    # Create dataframe using real header.
    # --------------------------------------------------------

    header = raw.iloc[
        header_row
    ].tolist()

    columns = []

    for idx, value in enumerate(
        header
    ):

        if pd.isna(value):

            columns.append(
                f"Unnamed_{idx}"
            )

        else:

            columns.append(
                str(value).strip()
            )

    df = raw.iloc[
        header_row + 1 :
    ].copy()

    df.columns = columns

    # --------------------------------------------------------
    # Find sample number and group columns.
    # --------------------------------------------------------

    sample_col = None
    group_col = None

    for col in df.columns:

        normalized = (
            str(col)
            .strip()
            .lower()
        )

        if normalized == "number":
            sample_col = col

        if normalized == "group":
            group_col = col

    if sample_col is None:

        raise ValueError(
            "Sample number column not found "
            f"in sheet: {sheet_name}"
        )

    if group_col is None:

        raise ValueError(
            "Group column not found "
            f"in sheet: {sheet_name}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Time is stored in COLUMN NAMES.
    #
    # Example:
    # 0.016667
    # 0.033333
    # 0.05
    # ...
    # 30.0
    # --------------------------------------------------------

    time_columns = []

    for col in df.columns:

        try:

            time_value = float(
                str(col).strip()
            )

        except (
            ValueError,
            TypeError,
        ):

            continue

        if np.isfinite(
            time_value
        ):

            time_columns.append(
                (
                    col,
                    time_value,
                )
            )

    if len(time_columns) < 10:

        raise ValueError(
            "Too few time-point columns "
            f"detected in {sheet_name}: "
            f"{len(time_columns)}"
        )

    # --------------------------------------------------------
    # Sort by retention time.
    # --------------------------------------------------------

    time_columns.sort(
        key=lambda x: x[1]
    )

    time = np.array(
        [
            value
            for _, value
            in time_columns
        ],
        dtype=float,
    )

    signal_columns = [
        col
        for col, _
        in time_columns
    ]

    # --------------------------------------------------------
    # Basic time diagnostics.
    # --------------------------------------------------------

    print(
        f"  Time points: {len(time)}"
    )

    print(
        f"  Time range : "
        f"{time.min():.6f} -> "
        f"{time.max():.6f} min"
    )

    # --------------------------------------------------------
    # Extract sample chromatograms.
    # --------------------------------------------------------

    signals = {}

    for _, row in df.iterrows():

        sample_id = extract_sample_id(
            row[sample_col]
        )

        if sample_id is None:
            continue

        y = pd.to_numeric(
            row[signal_columns],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite = np.isfinite(y)

        if finite.sum() < 2:
            continue

        # ----------------------------------------------------
        # Fill missing values by interpolation.
        # ----------------------------------------------------

        if not finite.all():

            y = np.interp(
                time,
                time[finite],
                y[finite],
            )

        signals[
            int(sample_id)
        ] = (
            time.copy(),
            y.copy(),
        )

    return signals


# ============================================================
# BUILD GROUP CENTROID
# ============================================================

def build_centroid(
    sample_ids,
    signals,
    reference_time,
):

    curves = []

    for sample_id in sample_ids:

        if sample_id not in signals:
            continue

        time, signal = (
            signals[sample_id]
        )

        curve = (
            interpolate_to_reference(
                time,
                snv(signal),
                reference_time,
            )
        )

        curves.append(curve)

    if not curves:
        return None

    return np.mean(
        np.vstack(curves),
        axis=0,
    )


# ============================================================
# LEAVE-ONE-OUT CENTROID
# ============================================================

def leave_one_out_centroid(
    group_sample_ids,
    sample_id,
    signals,
    reference_time,
):

    remaining_ids = [
        sid
        for sid in group_sample_ids
        if sid != sample_id
    ]

    return build_centroid(
        remaining_ids,
        signals,
        reference_time,
    )


# ============================================================
# ONE PREDICTION PER SAMPLE
# ============================================================

def select_prediction_per_sample(
    predictions,
):

    rows = []

    for sample_id, group in (
        predictions.groupby(
            "SampleId"
        )
    ):

        actual = (
            group[
                "ActualClass"
            ].iloc[0]
        )

        prediction_counts = (
            group[
                "PredictedClass"
            ]
            .value_counts()
        )

        predicted = (
            prediction_counts
            .index[0]
        )

        stability = (
            prediction_counts
            .iloc[0]
            / len(group)
        )

        rows.append(
            {
                "SampleId":
                    int(sample_id),

                "ActualClass":
                    actual,

                "PredictedClass":
                    predicted,

                "PredictionStability":
                    float(stability),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 78
    )
    print(
        "CANDIDATE 10 PROBLEM SAMPLE PROFILE"
    )
    print(
        "=" * 78
    )

    # ========================================================
    # INPUT CHECKS
    # ========================================================

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found:\n"
            f"{EXCEL_FILE}"
        )

    if not MAPPING_FILE.exists():

        raise FileNotFoundError(
            f"Mapping file not found:\n"
            f"{MAPPING_FILE}"
        )

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    if not STABLE_ERRORS_FILE.exists():

        raise FileNotFoundError(
            f"Stable errors file not found:\n"
            f"{STABLE_ERRORS_FILE}"
        )

    # ========================================================
    # LOAD MAPPING
    # ========================================================

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    sample_col = find_column(
        mapping,
        [
            "SampleId",
            "Sample ID",
            "Sample_ID",
            "sample",
            "id",
            "number",
        ],
    )

    group_col = find_column(
        mapping,
        [
            "Group",
            "GroupId",
            "Group ID",
            "Region",
            "Class",
        ],
    )

    if (
        sample_col is None
        or group_col is None
    ):

        raise ValueError(
            "Could not detect SampleId "
            "and Group columns in "
            "sample_mapping.csv"
        )

    mapping = mapping.copy()

    mapping["SampleId"] = (
        mapping[sample_col]
        .apply(extract_sample_id)
    )

    mapping["Group"] = (
        mapping[group_col]
        .astype(str)
        .str.strip()
    )

    mapping = mapping[
        mapping["SampleId"].notna()
    ].copy()

    mapping["SampleId"] = (
        mapping["SampleId"]
        .astype(int)
    )

    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    stable_errors = pd.read_csv(
        STABLE_ERRORS_FILE
    )

    selected = (
        select_prediction_per_sample(
            predictions
        )
    )

    stable_ids = (
        stable_errors[
            "SampleId"
        ]
        .astype(int)
        .tolist()
    )

    selected = selected[
        selected["SampleId"].isin(
            stable_ids
        )
    ].copy()

    selected = selected.merge(
        mapping[
            [
                "SampleId",
                "Group",
            ]
        ],
        on="SampleId",
        how="left",
    )

    # Authoritative mapping
    selected["ActualClass"] = (
        selected["Group"]
    )

    # ========================================================
    # PRINT SELECTED SAMPLES
    # ========================================================

    print()
    print(
        "Selected problematic samples:"
    )

    print(
        selected[
            [
                "SampleId",
                "ActualClass",
                "PredictedClass",
                "PredictionStability",
            ]
        ]
        .sort_values(
            "SampleId"
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LOAD THREE WAVELENGTHS
    # ========================================================

    all_signals = {}

    print()

    for wavelength, sheet_name in (
        WAVELENGTH_SHEETS.items()
    ):

        print(
            f"Loading {wavelength} nm..."
        )

        all_signals[wavelength] = (
            load_wavelength(
                sheet_name
            )
        )

        print(
            f"  Samples loaded: "
            f"{len(all_signals[wavelength])}"
        )

    # ========================================================
    # GROUP -> SAMPLE IDS
    # ========================================================

    group_to_samples = (
        mapping
        .groupby("Group")[
            "SampleId"
        ]
        .apply(list)
        .to_dict()
    )

    # ========================================================
    # PROFILE FEATURES
    # ========================================================

    profile_rows = []

    for _, row in selected.iterrows():

        sample_id = int(
            row["SampleId"]
        )

        actual = row["ActualClass"]
        predicted = row["PredictedClass"]
        stability = float(
            row["PredictionStability"]
        )

        for wavelength, signals in (
            all_signals.items()
        ):

            if sample_id not in signals:
                continue

            time, signal = (
                signals[sample_id]
            )

            # ------------------------------------------------
            # Signal statistics
            # ------------------------------------------------

            auc = float(
                np.trapezoid(
                    signal,
                    time,
                )
            )

            max_idx = int(
                np.argmax(signal)
            )

            min_idx = int(
                np.argmin(signal)
            )

            max_value = float(
                signal[max_idx]
            )

            min_value = float(
                signal[min_idx]
            )

            profile_rows.append(
                {
                    "SampleId":
                        sample_id,

                    "ActualClass":
                        actual,

                    "PredictedClass":
                        predicted,

                    "PredictionStability":
                        stability,

                    "Wavelength":
                        wavelength,

                    "AUC":
                        auc,

                    "Mean":
                        float(
                            np.mean(signal)
                        ),

                    "Std":
                        float(
                            np.std(signal)
                        ),

                    "Min":
                        min_value,

                    "Max":
                        max_value,

                    "MaxTime":
                        float(
                            time[max_idx]
                        ),

                    "MinTime":
                        float(
                            time[min_idx]
                        ),

                    "Range":
                        float(
                            max_value
                            - min_value
                        ),
                }
            )

    profile_df = pd.DataFrame(
        profile_rows
    )

    profile_df.to_csv(
        PROFILE_FILE,
        index=False,
    )

    # ========================================================
    # SIMILARITY ANALYSIS
    # ========================================================

    similarity_rows = []

    for _, row in selected.iterrows():

        sample_id = int(
            row["SampleId"]
        )

        actual = row["ActualClass"]
        predicted = row["PredictedClass"]

        for wavelength, signals in (
            all_signals.items()
        ):

            if sample_id not in signals:
                continue

            time, signal = (
                signals[sample_id]
            )

            # ------------------------------------------------
            # Reference grid.
            # Time is now guaranteed to be the real
            # chromatographic time axis from column names.
            # ------------------------------------------------

            reference_time = np.linspace(
                float(
                    np.min(time)
                ),
                float(
                    np.max(time)
                ),
                len(time),
            )

            sample_curve = (
                interpolate_to_reference(
                    time,
                    snv(signal),
                    reference_time,
                )
            )

            # ------------------------------------------------
            # Actual group centroid
            # Leave-one-out
            # ------------------------------------------------

            actual_ids = (
                group_to_samples.get(
                    actual,
                    [],
                )
            )

            actual_centroid = (
                leave_one_out_centroid(
                    actual_ids,
                    sample_id,
                    signals,
                    reference_time,
                )
            )

            # ------------------------------------------------
            # Predicted group centroid
            # Includes the sample itself.
            # ------------------------------------------------

            predicted_ids = (
                group_to_samples.get(
                    predicted,
                    [],
                )
            )

            predicted_centroid = (
                build_centroid(
                    predicted_ids,
                    signals,
                    reference_time,
                )
            )

            sim_actual = np.nan
            sim_predicted = np.nan

            if (
                actual_centroid
                is not None
            ):

                sim_actual = (
                    pearson_similarity(
                        sample_curve,
                        actual_centroid,
                    )
                )

            if (
                predicted_centroid
                is not None
            ):

                sim_predicted = (
                    pearson_similarity(
                        sample_curve,
                        predicted_centroid,
                    )
                )

            gap = np.nan

            if (
                np.isfinite(
                    sim_actual
                )
                and np.isfinite(
                    sim_predicted
                )
            ):

                gap = (
                    sim_predicted
                    - sim_actual
                )

            similarity_rows.append(
                {
                    "SampleId":
                        sample_id,

                    "ActualClass":
                        actual,

                    "PredictedClass":
                        predicted,

                    "Wavelength":
                        wavelength,

                    "SimilarityToActual":
                        sim_actual,

                    "SimilarityToPredicted":
                        sim_predicted,

                    "SimilarityGap_PredictedMinusActual":
                        gap,
                }
            )

    similarity_df = pd.DataFrame(
        similarity_rows
    )

    similarity_df.to_csv(
        SIMILARITY_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY PROFILE
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "PROFILE"
    )
    print(
        "=" * 78
    )

    if profile_df.empty:

        print(
            "No profile rows were generated."
        )

    else:

        print(
            profile_df
            .sort_values(
                [
                    "SampleId",
                    "Wavelength",
                ]
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.5f}",
            )
        )

    # ========================================================
    # DISPLAY SIMILARITY
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "SIMILARITY"
    )
    print(
        "=" * 78
    )

    if similarity_df.empty:

        print(
            "No similarity rows were generated."
        )

    else:

        print(
            similarity_df
            .sort_values(
                [
                    "SampleId",
                    "Wavelength",
                ]
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.5f}",
            )
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
        f"Profile    : {PROFILE_FILE}"
    )

    print(
        f"Similarity : {SIMILARITY_FILE}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()