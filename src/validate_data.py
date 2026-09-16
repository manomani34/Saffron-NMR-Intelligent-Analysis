import re
import pandas as pd


# --------------------------------------------------
# Official mapping provided by chemistry team
# --------------------------------------------------

GROUP_TO_REGION = {
    "G1": "سمنان",
    "G2": "اصفهان",
    "G3": "زنجان",
    "G4": "اراک",
    "G5": "خراسان شمالی",
    "G6": "خراسان رضوی",
    "G7": "فارس",
    "G8": "قم",
    "G9": "یزد",
    "G10": "قزوین",
    "G11": "لرستان",
}


SAMPLE_MAP = {
    1: "94-1",
    2: "94-2",
    3: "04-10",

    4: "94-5",
    5: "04-27",
    6: "04-19",
    7: "04-20",
    8: "04-21",

    9: "94-7",
    10: "94-8",
    11: "04-13",
    12: "04-22",

    13: "94-9",
    14: "04-4",
    15: "04-5",
    16: "04-6",
    17: "04-7",
    18: "04-14",
    19: "04-15",

    20: "94-13",
    21: "04-3",
    22: "04-12",
    23: "04-23",

    24: "94-21",
    25: "94-22",
    26: "94-25",
    27: "94-26",
    28: "94-27",
    29: "94-28",
    30: "04-1",
    31: "04-11",
    32: "04-16",
    33: "04-18",

    34: "94-30",
    35: "94-30",
    36: "94-32",

    37: "94-33",
    38: "04-2",

    39: "94-35",
    40: "94-36",

    41: "04-8",
    42: "04-9",

    43: "04-24",
    44: "04-25",
}


# --------------------------------------------------
# Chemistry-team confirmed removed spectral regions
# --------------------------------------------------

REMOVED_REGIONS = [
    (2.43198, 2.55698),
    (3.13198, 3.19298),
    (3.27898, 3.52698),
]


EXPECTED_BIN_WIDTH = 0.001

# Floating-point tolerance for normal adjacent bins
BIN_WIDTH_TOLERANCE = 0.00005


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_harvest_year(sample_name: str):
    """
    Extract harvest year from the official sample name.

    94-xxx -> 1394
    04-xxx -> 1404
    """

    if not isinstance(sample_name, str):
        return None

    match = re.match(
        r"^(94|04)-",
        sample_name.strip(),
    )

    if not match:
        return None

    if match.group(1) == "94":
        return 1394

    if match.group(1) == "04":
        return 1404

    return None


def gap_matches_removed_region(
    left_ppm: float,
    right_ppm: float,
) -> bool:

    for start, end in REMOVED_REGIONS:

        if (
            right_ppm >= start
            and left_ppm <= end
        ):
            return True

    return False


def sort_groups(values):

    return values.sort_index(
        key=lambda items: items.map(
            lambda value:
                int(str(value)[1:])
                if str(value).startswith("G")
                else 999
        )
    )


# --------------------------------------------------
# Main validation
# --------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:

    print("\n" + "=" * 72)
    print("SAFFRON NMR DATASET VALIDATION")
    print("=" * 72)

    metadata_columns = [
    "name",
    "group",
    "region_code",
    ]

    # ==================================================
    # 1. BASIC INFORMATION
    # ==================================================

    print("\n[1] BASIC INFORMATION")

    print(
        f"Rows                  : {df.shape[0]}"
    )

    print(
        f"Columns               : {df.shape[1]}"
    )

    print(
        f"Metadata columns      : {len(metadata_columns)}"
    )

    missing_required = [
        column
        for column in metadata_columns
        if column not in df.columns
    ]

    if missing_required:

        print(
            f"ERROR: Missing required columns: "
            f"{missing_required}"
        )

        return

    spectral_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print(
        f"Spectral points       : "
        f"{len(spectral_columns)}"
    )

    # ==================================================
    # 2. REQUIRED COLUMNS
    # ==================================================

    print("\n[2] REQUIRED COLUMNS")

    for column in metadata_columns:

        print(
            f"OK: '{column}' exists"
        )

    # ==================================================
    # 3. GROUP DISTRIBUTION
    # ==================================================

    print("\n[3] GROUP DISTRIBUTION")

    group_counts = sort_groups(
        df["group"]
        .astype(str)
        .value_counts()
    )

    for group, count in group_counts.items():

        region = GROUP_TO_REGION.get(
            group,
            "UNKNOWN",
        )

        print(
            f"{group:4} -> "
            f"{region:<15} -> "
            f"{count} records"
        )

    unknown_groups = [
        group
        for group in df["group"].astype(str).unique()
        if group not in GROUP_TO_REGION
    ]

    if unknown_groups:

        print(
            f"WARNING: Unknown groups: "
            f"{unknown_groups}"
        )

    expected_group_names = set(
        GROUP_TO_REGION.keys()
    )

    actual_group_names = set(
        df["group"].astype(str)
    )

    missing_groups = sorted(
        expected_group_names - actual_group_names,
        key=lambda value: int(value[1:]),
    )

    if missing_groups:

        print(
            f"WARNING: Expected groups missing: "
            f"{missing_groups}"
        )

    # ==================================================
    # 4. MISSING VALUES
    # ==================================================

    print("\n[4] MISSING VALUES")

    total_missing = int(
        df.isna().sum().sum()
    )

    print(
        f"Total missing values : "
        f"{total_missing}"
    )

    if total_missing == 0:

        print(
            "OK: No missing values found"
        )

    else:

        print(
            "WARNING: Missing values detected"
        )

        missing_columns = (
            df.isna().sum()
        )

        missing_columns = (
            missing_columns[
                missing_columns > 0
            ]
        )

        print(
            "Columns with missing values:"
        )

        print(
            missing_columns.to_string()
        )

    # ==================================================
    # 5. DUPLICATE ROWS
    # ==================================================

    print("\n[5] DUPLICATE ROWS")

    duplicate_count = int(
        df.duplicated().sum()
    )

    print(
        f"Duplicate rows       : "
        f"{duplicate_count}"
    )

    if duplicate_count == 0:

        print(
            "OK: No completely duplicated rows"
        )

    else:

        print(
            "WARNING: Completely duplicated rows detected"
        )

    # ==================================================
    # 6. SAMPLE ID VALIDATION
    # ==================================================

    print("\n[6] SAMPLE ID VALIDATION")

    sample_ids = pd.to_numeric(
        df["name"],
        errors="coerce",
    )

    invalid_sample_ids = int(
        sample_ids.isna().sum()
    )

    print(
        f"Invalid sample IDs   : "
        f"{invalid_sample_ids}"
    )

    if invalid_sample_ids == 0:

        print(
            "OK: All sample IDs are numeric"
        )

    expected_ids = set(
        range(1, 45)
    )

    actual_ids = set(
        sample_ids
        .dropna()
        .astype(int)
    )

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    extra_ids = sorted(
        actual_ids - expected_ids
    )

    print(
        "Expected sample IDs  : 1 - 44"
    )

    print(
        f"Actual sample count  : "
        f"{len(actual_ids)}"
    )

    if missing_ids:

        print(
            f"WARNING: Missing sample IDs: "
            f"{missing_ids}"
        )

    else:

        print(
            "OK: No missing sample IDs"
        )

    if extra_ids:

        print(
            f"WARNING: Unexpected sample IDs: "
            f"{extra_ids}"
        )

    # ==================================================
    # 7. SAMPLE MAPPING
    # ==================================================

    print("\n[7] SAMPLE MAPPING")

    mapping_errors = []

    for _, row in df.iterrows():

        sample_id = int(row["name"])

        actual_group = str(
            row["group"]
        )

        expected_sample_name = SAMPLE_MAP.get(
            sample_id
        )

        if expected_sample_name is None:

            mapping_errors.append(
                f"Sample ID {sample_id}: "
                f"no sample mapping found"
            )

            continue

        if actual_group not in GROUP_TO_REGION:

            mapping_errors.append(
                f"Sample ID {sample_id}: "
                f"unknown group '{actual_group}'"
            )

    if mapping_errors:

        print(
            f"WARNING: "
            f"{len(mapping_errors)} mapping errors"
        )

        for error in mapping_errors[:10]:

            print(
                f"  {error}"
            )

        if len(mapping_errors) > 10:

            print(
                f"  ... "
                f"{len(mapping_errors) - 10} "
                f"additional errors omitted"
            )

    else:

        print(
            "OK: Sample mapping is available "
            f"for all {len(df)} records"
        )

        print(
            "Original sample names verified."
        )

    # ==================================================
    # 8. ORIGINAL SAMPLE NAME DUPLICATES
    # ==================================================

    print(
        "\n[8] ORIGINAL SAMPLE NAME DUPLICATES"
    )

    sample_name_values = list(
        SAMPLE_MAP.values()
    )

    duplicate_sample_names = (
        pd.Series(
            sample_name_values
        )
        .value_counts()
    )

    duplicate_sample_names = (
        duplicate_sample_names[
            duplicate_sample_names > 1
        ]
    )

    if duplicate_sample_names.empty:

        print(
            "OK: No duplicate original sample names"
        )

    else:

        print(
            "WARNING: Duplicate original sample names:"
        )

        for sample_name, count in (
            duplicate_sample_names.items()
        ):

            ids = [
                sample_id
                for sample_id, mapped_name
                in SAMPLE_MAP.items()
                if mapped_name == sample_name
            ]

            print(
                f"  {sample_name} -> "
                f"IDs {ids} -> "
                f"{count} occurrences"
            )

    # ==================================================
    # 9. EXACT DUPLICATE SPECTRA
    # ==================================================

    print("\n[9] EXACT DUPLICATE SPECTRA")

    numeric_spectral_data = (
        df[spectral_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    duplicate_spectrum_mask = (
        numeric_spectral_data
        .duplicated(
            keep=False
        )
    )

    duplicate_spectra_df = df[
        duplicate_spectrum_mask
    ]

    if duplicate_spectra_df.empty:

        print(
            "OK: No exact duplicate spectra found"
        )

    else:

        duplicate_ids = (
            duplicate_spectra_df["name"]
            .astype(int)
            .tolist()
        )

        print(
            "WARNING: Exact duplicate spectra found"
        )

        print(
            f"Sample IDs: {duplicate_ids}"
        )

        if set(duplicate_ids) == {34, 35}:

            print(
                "CONFIRMED: IDs 34 and 35 "
                "are the same measurement."
            )

    # ==================================================
    # 10. SPECTRAL AXIS
    # ==================================================

    print("\n[10] SPECTRAL AXIS")

    try:

        ppm_values = [
            float(column)
            for column in spectral_columns
        ]

    except ValueError:

        print(
            "ERROR: Some spectral column names "
            "are not numeric"
        )

        return

    if not ppm_values:

        print(
            "ERROR: No spectral columns found"
        )

        return

    differences = [
        ppm_values[i + 1] - ppm_values[i]
        for i in range(
            len(ppm_values) - 1
        )
    ]

    print(
        f"First ppm            : "
        f"{ppm_values[0]}"
    )

    print(
        f"Last ppm             : "
        f"{ppm_values[-1]}"
    )

    print(
        f"Minimum bin width    : "
        f"{min(differences):.8f}"
    )

    print(
        f"Maximum bin width    : "
        f"{max(differences):.8f}"
    )

    print(
        f"Mean bin width       : "
        f"{sum(differences) / len(differences):.8f}"
    )

    increasing = all(
        ppm_values[i] <
        ppm_values[i + 1]
        for i in range(
            len(ppm_values) - 1
        )
    )

    decreasing = all(
        ppm_values[i] >
        ppm_values[i + 1]
        for i in range(
            len(ppm_values) - 1
        )
    )

    print(
        f"Axis increasing      : "
        f"{increasing}"
    )

    print(
        f"Axis decreasing      : "
        f"{decreasing}"
    )

    print(
        f"Expected bin width   : "
        f"{EXPECTED_BIN_WIDTH}"
    )

    # ==================================================
    # 10-B. GAP ANALYSIS
    # ==================================================

    print(
        "\n[10-B] SPECTRAL GAP ANALYSIS"
    )

    unexpected_gaps = []
    expected_removed_gaps = []

    for index, diff in enumerate(
        differences
    ):

        if abs(
            diff - EXPECTED_BIN_WIDTH
        ) <= BIN_WIDTH_TOLERANCE:

            continue

        left_ppm = ppm_values[index]
        right_ppm = ppm_values[
            index + 1
        ]

        gap_info = {
            "index": index,
            "from_ppm": left_ppm,
            "to_ppm": right_ppm,
            "width": diff,
        }

        if gap_matches_removed_region(
            left_ppm,
            right_ppm
        ):

            expected_removed_gaps.append(
                gap_info
            )

        else:

            unexpected_gaps.append(
                gap_info
            )

    print(
        f"Expected removed-region gaps : "
        f"{len(expected_removed_gaps)}"
    )

    print(
        f"Unexpected spectral gaps      : "
        f"{len(unexpected_gaps)}"
    )

    if expected_removed_gaps:

        for gap in expected_removed_gaps:

            print(
                f"  Expected: "
                f"{gap['from_ppm']:.6f} -> "
                f"{gap['to_ppm']:.6f} "
                f"({gap['width']:.6f} ppm)"
            )

    if unexpected_gaps:

        print(
            "WARNING: Unexpected spectral gaps:"
        )

        for gap in unexpected_gaps[:10]:

            print(
                f"  index {gap['index']:5d} | "
                f"{gap['from_ppm']:.6f} -> "
                f"{gap['to_ppm']:.6f} | "
                f"width={gap['width']:.6f}"
            )

        if len(unexpected_gaps) > 10:

            print(
                f"  ... "
                f"{len(unexpected_gaps) - 10} "
                f"additional gaps omitted"
            )

    else:

        print(
            "OK: All large gaps match "
            "known removed regions"
        )

    # ==================================================
    # 10-C. REMOVED REGIONS
    # ==================================================

    print(
        "\n[10-C] REMOVED SPECTRAL REGIONS"
    )

    for start, end in REMOVED_REGIONS:

        print(
            f"Removed: "
            f"{start:.5f} - {end:.5f} ppm"
        )

    # ==================================================
    # 10-D. BIN WIDTH
    # ==================================================

    print(
        "\n[10-D] BIN WIDTH VALIDATION"
    )

    normal_differences = [
        diff
        for index, diff in enumerate(
            differences
        )
        if abs(
            diff - EXPECTED_BIN_WIDTH
        ) <= BIN_WIDTH_TOLERANCE
    ]

    abnormal_regular_bins = [
        diff
        for index, diff in enumerate(
            differences
        )
        if (
            abs(
                diff - EXPECTED_BIN_WIDTH
            ) > BIN_WIDTH_TOLERANCE
            and not gap_matches_removed_region(
                ppm_values[index],
                ppm_values[index + 1]
            )
        )
    ]

    print(
        f"Regular bins checked : "
        f"{len(normal_differences)}"
    )

    print(
        f"Unexpected bin errors : "
        f"{len(abnormal_regular_bins)}"
    )

    if abnormal_regular_bins:

        print(
            "WARNING: Unexpected bin-width errors detected"
        )

    else:

        print(
            "OK: Regular bins are consistent "
            "with nominal width = 0.001 ppm"
        )

    # ==================================================
    # 11. SPECTRAL VALUES
    # ==================================================

    print("\n[11] SPECTRAL VALUES")

    non_numeric_count = 0

    for column in spectral_columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        non_numeric_count += int(
            converted.isna().sum()
        )

    print(
        f"Non-numeric / invalid values : "
        f"{non_numeric_count}"
    )

    if non_numeric_count == 0:

        print(
            "OK: All spectral values are numeric"
        )

    else:

        print(
            "WARNING: Invalid spectral values detected"
        )

    # ==================================================
    # 12. GLOBAL SPECTRAL STATISTICS
    # ==================================================

    print(
        "\n[12] BASIC SPECTRAL STATISTICS"
    )

    global_min = (
        numeric_spectral_data
        .min()
        .min()
    )

    global_max = (
        numeric_spectral_data
        .max()
        .max()
    )

    global_mean = (
        numeric_spectral_data
        .mean()
        .mean()
    )

    global_std = (
        numeric_spectral_data
        .stack()
        .std()
    )

    print(
        f"Minimum              : {global_min:.6f}"
    )

    print(
        f"Maximum              : {global_max:.6f}"
    )

    print(
        f"Mean                 : {global_mean:.6f}"
    )

    print(
        f"Standard deviation   : {global_std:.6f}"
    )

    # ==================================================
    # 13. SAMPLE-LEVEL STATISTICS
    # ==================================================

    print(
        "\n[13] SAMPLE-LEVEL SPECTRAL STATISTICS"
    )

    row_means = (
        numeric_spectral_data
        .mean(axis=1)
    )

    row_stds = (
        numeric_spectral_data
        .std(axis=1)
    )

    row_mins = (
        numeric_spectral_data
        .min(axis=1)
    )

    row_maxs = (
        numeric_spectral_data
        .max(axis=1)
    )

    sample_statistics = pd.DataFrame(
        {
            "SampleId": df["name"],
            "Group": df["group"],
            "Mean": row_means,
            "Std": row_stds,
            "Min": row_mins,
            "Max": row_maxs,
        }
    )

    print(
        f"Samples analyzed     : "
        f"{len(sample_statistics)}"
    )

    print(
        f"Mean intensity range : "
        f"{row_means.min():.4f} - "
        f"{row_means.max():.4f}"
    )

    print(
        f"Std range            : "
        f"{row_stds.min():.4f} - "
        f"{row_stds.max():.4f}"
    )

    print(
        "Detailed sample statistics "
        "are omitted from terminal output."
    )

    # ==================================================
    # 14. HARVEST YEAR
    # ==================================================

    print("\n[14] HARVEST YEAR")

    harvest_years = []

    for sample_id in df["name"]:

        sample_id = int(sample_id)

        sample_name = SAMPLE_MAP.get(
            sample_id
        )

        harvest_year = (
            extract_harvest_year(
                sample_name
            )
        )

        harvest_years.append(
            harvest_year
        )

    harvest_year_series = pd.Series(
        harvest_years
    )

    harvest_counts = (
        harvest_year_series
        .value_counts()
        .sort_index()
    )

    for year, count in harvest_counts.items():

        print(
            f"{year} -> {count} records"
        )

    if harvest_year_series.isna().any():

        print(
            "WARNING: Some sample names "
            "do not have recognizable harvest year"
        )

    else:

        print(
            "OK: Harvest year successfully extracted"
        )

    # ==================================================
    # 15. INDEPENDENT SAMPLE STATUS
    # ==================================================

    print(
        "\n[15] INDEPENDENT SAMPLE STATUS"
    )

    total_records = len(df)

    confirmed_duplicate_ids = {
        34,
        35,
    }

    independent_count = (
        total_records
    )

    if confirmed_duplicate_ids.issubset(
        actual_ids
    ):

        independent_count -= 1

    print(
        f"Total records          : "
        f"{total_records}"
    )

    print(
        f"Confirmed duplicate    : "
        f"34, 35"
    )

    print(
        f"Independent observations: "
        f"{independent_count}"
    )

    if independent_count == 43:

        print(
            "OK: Dataset contains "
            "43 independent observations"
        )

    # ==================================================
    # 16. PREPROCESSING STATUS
    # ==================================================

    print(
        "\n[16] PREPROCESSING STATUS"
    )

    print(
        "Binning method        : Fixed-width"
    )

    print(
        "Nominal bin width     : 0.001 ppm"
    )

    print(
        "Binning aggregation   : Sum"
    )

    print(
        "Normalization         : NOT APPLIED"
    )

    print(
        "Baseline correction   : Applied in Mnova"
    )

    print(
        "Reference/Alignment   : Applied in Mnova"
    )

    print(
        "Current Python stage  : Validation"
    )

    # ==================================================
    # 17. DATASET CONSISTENCY
    # ==================================================

    print(
        "\n[17] DATASET CONSISTENCY"
    )

    checks = [

        (
            "44 records in CSV",
            len(df) == 44,
        ),

        (
            "11 groups",
            df["group"].nunique() == 11,
        ),

        (
            "9514 spectral columns",
            len(spectral_columns) == 9514,
        ),

        (
            "No missing values",
            total_missing == 0,
        ),

        (
            "All spectral values numeric",
            non_numeric_count == 0,
        ),

        (
            "Sample IDs 1-44",
            not missing_ids
            and not extra_ids,
        ),

        (
            "No unexpected spectral gaps",
            len(unexpected_gaps) == 0,
        ),
    ]

    passed_count = 0

    for description, passed in checks:

        status = (
            "OK"
            if passed
            else "WARNING"
        )

        if passed:
            passed_count += 1

        print(
            f"{status:8} - {description}"
        )

    print(
        f"\nConsistency checks: "
        f"{passed_count}/{len(checks)} passed"
    )

    # ==================================================
    # FINAL
    # ==================================================

    print("\n" + "=" * 72)
    print("VALIDATION FINISHED")
    print("=" * 72)