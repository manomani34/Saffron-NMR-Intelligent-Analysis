from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_DATASET_PATH = Path("data/raw/color.csv")
DEFAULT_OUTPUT_PATH = Path("color_dataset_audit.json")

SOLVENT_REGION = (4.66797, 5.49960)

ARTIFICIAL_COLOR_REGION = (5.0, 9.0)

EXPECTED_BIN_WIDTH = 0.001
BIN_WIDTH_TOLERANCE = 0.00005

EMPTY_COLUMN_PREFIX = "Unnamed:"


# ============================================================
# HELPERS
# ============================================================

def _is_float_column_name(value):
    try:
        float(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False


def _is_empty_column(column):
    return str(column).startswith(EMPTY_COLUMN_PREFIX)


def _extract_spectral_columns(df):
    """
    Spectral columns are columns whose names are valid numeric ppm values.
    Empty 'Unnamed:*' columns are intentionally ignored.
    """

    columns = []

    for column in df.columns:
        if _is_float_column_name(column):
            columns.append(column)

    return columns


def _extract_metadata_columns(df):
    """
    Metadata columns are non-numeric column names that actually contain
    information.

    Completely empty Unnamed columns are ignored.
    """

    metadata_columns = []

    for column in df.columns:

        if _is_float_column_name(column):
            continue

        if _is_empty_column(column):
            continue

        metadata_columns.append(column)

    return metadata_columns


def _find_empty_columns(df):
    empty_columns = []

    for column in df.columns:
        if df[column].isna().all():
            empty_columns.append(column)

    return empty_columns


def _file_sha256(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def _parse_ppm_columns(columns):
    ppm = []

    for column in columns:
        try:
            ppm.append(float(str(column).strip()))
        except (ValueError, TypeError):
            pass

    return np.asarray(ppm, dtype=float)


def _analyze_axis(ppm):
    result = {
        "valid": False,
        "increasing": False,
        "decreasing": False,
        "min_ppm": None,
        "max_ppm": None,
        "mean_bin_width": None,
        "min_bin_width": None,
        "max_bin_width": None,
        "normal_bin_width_mean": None,
        "normal_bin_width_min": None,
        "normal_bin_width_max": None,
        "unexpected_gaps": [],
    }

    if len(ppm) < 2:
        return result

    differences = np.diff(ppm)
    absolute_differences = np.abs(differences)

    increasing = bool(np.all(differences > 0))
    decreasing = bool(np.all(differences < 0))

    result["valid"] = increasing or decreasing
    result["increasing"] = increasing
    result["decreasing"] = decreasing

    result["min_ppm"] = float(ppm.min())
    result["max_ppm"] = float(ppm.max())

    result["mean_bin_width"] = float(
        np.mean(absolute_differences)
    )

    result["min_bin_width"] = float(
        np.min(absolute_differences)
    )

    result["max_bin_width"] = float(
        np.max(absolute_differences)
    )

    normal_differences = []

    for left, right, diff in zip(
        ppm[:-1],
        ppm[1:],
        absolute_differences,
    ):
        if np.isclose(
            diff,
            EXPECTED_BIN_WIDTH,
            atol=BIN_WIDTH_TOLERANCE,
        ):
            normal_differences.append(diff)

            continue

        result["unexpected_gaps"].append(
            {
                "left_ppm": float(left),
                "right_ppm": float(right),
                "gap": float(diff),
            }
        )

    if normal_differences:
        result["normal_bin_width_mean"] = float(
            np.mean(normal_differences)
        )

        result["normal_bin_width_min"] = float(
            np.min(normal_differences)
        )

        result["normal_bin_width_max"] = float(
            np.max(normal_differences)
        )

    return result


def _count_columns_in_region(
    ppm,
    lower,
    upper,
):
    if len(ppm) == 0:
        return 0

    mask = (
        (ppm >= lower)
        & (ppm <= upper)
    )

    return int(mask.sum())


def _detect_solvent_gap(ppm):
    """
    Detect whether the solvent region is physically absent from the
    spectral axis.

    For color.csv the expected result is approximately:

        last point before solvent ~= 4.66697
        first point after solvent ~= 5.05096
    """

    if len(ppm) < 2:
        return {
            "detected": False,
            "last_before_region": None,
            "first_after_region": None,
            "gap": None,
        }

    before = ppm[
        ppm < SOLVENT_REGION[0]
    ]

    after = ppm[
        ppm > SOLVENT_REGION[1]
    ]

    if len(before) == 0 or len(after) == 0:
        return {
            "detected": False,
            "last_before_region": None,
            "first_after_region": None,
            "gap": None,
        }

    last_before = float(before[-1])
    first_after = float(after[0])

    gap = first_after - last_before

    detected = gap > 0.1

    return {
        "detected": detected,
        "last_before_region": last_before,
        "first_after_region": first_after,
        "gap": float(gap),
    }


# ============================================================
# DATASET AUDIT
# ============================================================

def audit_dataset(
    dataset_path=DEFAULT_DATASET_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    print("=" * 70)
    print("SAFFRON NMR COLOR DATASET AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    print()
    print(f"Dataset : {dataset_path}")

    print(
        f"Size    : "
        f"{dataset_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    file_hash = _file_sha256(
        dataset_path
    )

    print(
        f"SHA256  : {file_hash}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
    )

    rows = len(df)
    columns = len(df.columns)

    print()
    print("[1] BASIC INFORMATION")

    print(
        f"Rows    : {rows}"
    )

    print(
        f"Columns : {columns}"
    )

    # --------------------------------------------------------
    # COLUMN STRUCTURE
    # --------------------------------------------------------

    spectral_columns = (
        _extract_spectral_columns(df)
    )

    metadata_columns = (
        _extract_metadata_columns(df)
    )

    empty_columns = (
        _find_empty_columns(df)
    )

    ppm = _parse_ppm_columns(
        spectral_columns
    )

    print()
    print("[2] COLUMN STRUCTURE")

    print(
        f"Metadata columns      : "
        f"{len(metadata_columns)}"
    )

    print(
        f"Spectral points       : "
        f"{len(spectral_columns)}"
    )

    print(
        f"Completely empty cols : "
        f"{len(empty_columns)}"
    )

    print()
    print("Metadata:")

    for column in metadata_columns:
        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # REQUIRED COLOR DATASET COLUMNS
    # --------------------------------------------------------

    print()
    print("[3] COLOR DATASET METADATA")

    name_column = None

    for candidate in (
        "name",
        "name ",
        "name  ",
    ):
        if candidate in df.columns:
            name_column = candidate
            break

    number_column = None

    if "number" in df.columns:
        number_column = "number"

    print(
        f"Sample name column : "
        f"{name_column}"
    )

    print(
        f"Sample number column : "
        f"{number_column}"
    )

    metadata_ok = (
        name_column is not None
        and number_column is not None
    )

    print(
        f"Metadata structure : "
        f"{'PASS' if metadata_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # SAMPLE NAMES
    # --------------------------------------------------------

    print()
    print("[4] SAMPLES")

    sample_names = []

    if name_column is not None:

        sample_names = (
            df[name_column]
            .astype(str)
            .str.strip()
            .tolist()
        )

        for index, name in enumerate(
            sample_names,
            start=1,
        ):
            print(
                f"  {index:>2}. {name}"
            )

    print(
        f"Total samples : "
        f"{len(sample_names)}"
    )

    # --------------------------------------------------------
    # SAMPLE NUMBERS
    # --------------------------------------------------------

    print()
    print("[5] SAMPLE NUMBERS")

    invalid_numbers = 0
    duplicate_numbers = []

    if number_column is not None:

        numbers = pd.to_numeric(
            df[number_column],
            errors="coerce",
        )

        invalid_numbers = int(
            numbers.isna().sum()
        )

        duplicate_numbers = (
            numbers[
                numbers.duplicated(
                    keep=False
                )
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        print(
            f"Invalid numbers : "
            f"{invalid_numbers}"
        )

        print(
            f"Duplicate numbers : "
            f"{duplicate_numbers}"
        )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print()
    print("[6] MISSING VALUES")

    missing_by_column = (
        df.isna().sum()
    )

    real_missing_columns = (
        missing_by_column[
            (missing_by_column > 0)
            & ~missing_by_column.index.astype(str).str.startswith(
                EMPTY_COLUMN_PREFIX
            )
        ]
    )

    if real_missing_columns.empty:
        print(
            "No missing values in "
            "meaningful columns."
        )
    else:

        print(
            f"Meaningful columns with "
            f"missing values: "
            f"{len(real_missing_columns)}"
        )

        for column, count in (
            real_missing_columns
            .sort_values(
                ascending=False
            )
            .items()
        ):
            print(
                f"  {column}: {count}"
            )

    # --------------------------------------------------------
    # EMPTY COLUMNS
    # --------------------------------------------------------

    print()
    print("[7] EMPTY CSV COLUMNS")

    print(
        f"Completely empty columns : "
        f"{len(empty_columns)}"
    )

    if empty_columns:
        print(
            "These columns are ignored "
            "and are not metadata."
        )

    # --------------------------------------------------------
    # SPECTRAL DATA
    # --------------------------------------------------------

    print()
    print("[8] SPECTRAL DATA")

    X = (
        df[spectral_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    non_numeric = int(
        X.isna().sum().sum()
    )

    array = X.to_numpy(
        dtype=float
    )

    finite_mask = np.isfinite(
        array
    )

    non_finite = int(
        (~finite_mask).sum()
    )

    print(
        f"Non-numeric cells : "
        f"{non_numeric}"
    )

    print(
        f"Non-finite cells  : "
        f"{non_finite}"
    )

    # --------------------------------------------------------
    # SPECTRAL AXIS
    # --------------------------------------------------------

    print()
    print("[9] SPECTRAL AXIS")

    axis = _analyze_axis(
        ppm
    )

    if len(ppm) == 0:

        print(
            "ERROR: No spectral axis found."
        )

    else:

        print(
            f"PPM min : "
            f"{axis['min_ppm']:.6f}"
        )

        print(
            f"PPM max : "
            f"{axis['max_ppm']:.6f}"
        )

        print(
            f"Increasing : "
            f"{axis['increasing']}"
        )

        print(
            f"Decreasing : "
            f"{axis['decreasing']}"
        )

        print(
            f"Mean bin width : "
            f"{axis['mean_bin_width']:.8f}"
        )

        print(
            f"Normal bin width mean : "
            f"{axis['normal_bin_width_mean']:.8f}"
            if axis["normal_bin_width_mean"] is not None
            else
            "Normal bin width mean : N/A"
        )

        print(
            f"Unexpected gaps : "
            f"{len(axis['unexpected_gaps'])}"
        )

    # --------------------------------------------------------
    # SOLVENT REGION
    # --------------------------------------------------------

    print()
    print("[10] SOLVENT REGION")

    solvent_gap = _detect_solvent_gap(
        ppm
    )

    solvent_columns = _count_columns_in_region(
    ppm,
    SOLVENT_REGION[0],
    SOLVENT_REGION[1],
)

    solvent_gap_detected = solvent_gap["detected"]
    solvent_region_removed = solvent_gap_detected

    print(
        f"Expected solvent region : "
        f"{SOLVENT_REGION[0]} - "
        f"{SOLVENT_REGION[1]} ppm"
    )

    print(
    f"Numeric columns inside region : "
    f"{solvent_columns}"
)

    print(
    f"Solvent gap detected          : "
    f"{solvent_gap_detected}"
)

    print(
    f"Solvent region removed        : "
    f"{solvent_region_removed}"
)

    if solvent_gap["detected"]:

        print(
            f"Last point before region : "
            f"{solvent_gap['last_before_region']:.6f}"
        )

        print(
            f"First point after region : "
            f"{solvent_gap['first_after_region']:.6f}"
        )

        print(
            f"Observed gap             : "
            f"{solvent_gap['gap']:.6f}"
        )

    # --------------------------------------------------------
    # ARTIFICIAL COLOR REGION
    # --------------------------------------------------------

    print()
    print("[11] ARTIFICIAL COLOR REGION")

    color_region_columns = (
        _count_columns_in_region(
            ppm,
            ARTIFICIAL_COLOR_REGION[0],
            ARTIFICIAL_COLOR_REGION[1],
        )
    )

    print(
        f"Primary color region : "
        f"{ARTIFICIAL_COLOR_REGION[0]} - "
        f"{ARTIFICIAL_COLOR_REGION[1]} ppm"
    )

    print(
        f"Spectral points in region : "
        f"{color_region_columns}"
    )

    # --------------------------------------------------------
    # EXACT DUPLICATES
    # --------------------------------------------------------

    print()
    print("[12] DUPLICATE SPECTRA")

    duplicate_spectra = int(
        X.duplicated(
            keep="first"
        ).sum()
    )

    duplicate_rows = []

    if duplicate_spectra > 0:

        duplicate_mask = X.duplicated(
            keep=False
        )

        duplicate_rows = (
            df.index[
                duplicate_mask
            ]
            .tolist()
        )

    print(
        f"Exact duplicate spectra : "
        f"{duplicate_spectra}"
    )

    if duplicate_rows:

        print(
            f"Related row indexes : "
            f"{duplicate_rows}"
        )

        if name_column is not None:

            duplicate_names = (
                df.loc[
                    duplicate_mask,
                    name_column,
                ]
                .astype(str)
                .str.strip()
                .tolist()
            )

            print(
                f"Related sample names : "
                f"{duplicate_names}"
            )

    # --------------------------------------------------------
    # SPECTRAL STATISTICS
    # --------------------------------------------------------

    print()
    print("[13] SPECTRAL STATISTICS")

    if array.size > 0:

        print(
            f"Global minimum : "
            f"{np.nanmin(array):.8g}"
        )

        print(
            f"Global maximum : "
            f"{np.nanmax(array):.8g}"
        )

        print(
            f"Global mean    : "
            f"{np.nanmean(array):.8g}"
        )

        print(
            f"Global std     : "
            f"{np.nanstd(array):.8g}"
        )

    # --------------------------------------------------------
    # STRUCTURAL COMPATIBILITY
    # --------------------------------------------------------

    print()
    print("[14] STRUCTURAL COMPATIBILITY")

    compatibility = {
        "metadata_name_column": (
            name_column is not None
        ),

        "metadata_number_column": (
            number_column is not None
        ),

        "spectral_columns_found": (
            len(spectral_columns) > 0
        ),

        "spectral_values_numeric": (
            non_numeric == 0
        ),

        "spectral_values_finite": (
            non_finite == 0
        ),

        "spectral_axis_valid": (
            axis["valid"]
        ),

        "solvent_region_removed": (
    solvent_region_removed
        ),
    }

    for key, value in compatibility.items():

        status = (
            "PASS"
            if value
            else "FAIL"
        )

        print(
            f"  [{status}] {key}"
        )

    structural_checks = [
        compatibility[
            "metadata_name_column"
        ],

        compatibility[
            "metadata_number_column"
        ],

        compatibility[
            "spectral_columns_found"
        ],

        compatibility[
            "spectral_values_numeric"
        ],

        compatibility[
            "spectral_values_finite"
        ],

        compatibility[
            "spectral_axis_valid"
        ],

        compatibility[
            "solvent_region_removed"
        ],
    ]

    overall_valid = all(
        structural_checks
    )

    print()
    print("=" * 70)

    if overall_valid:

        print(
            "RESULT: COLOR DATASET "
            "IS STRUCTURALLY VALID"
        )

    else:

        print(
            "RESULT: COLOR DATASET "
            "REQUIRES REVIEW"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "dataset": {
            "path": str(
                dataset_path
            ),
            "filename":
                dataset_path.name,
            "size_bytes":
                dataset_path.stat().st_size,
            "sha256":
                file_hash,
            "rows":
                rows,
            "columns":
                columns,
        },

        "structure": {
            "metadata_columns":
                metadata_columns,
            "spectral_column_count":
                len(spectral_columns),
            "empty_column_count":
                len(empty_columns),
        },

        "metadata": {
            "name_column":
                name_column,
            "number_column":
                number_column,
            "metadata_valid":
                metadata_ok,
        },

        "samples": {
            "count":
                len(sample_names),
            "names":
                sample_names,
            "invalid_numbers":
                invalid_numbers,
            "duplicate_numbers":
                duplicate_numbers,
        },

        "spectral_axis": {
            **axis,
        },

        "solvent_region": {
            "expected_lower_ppm":
                SOLVENT_REGION[0],
            "expected_upper_ppm":
                SOLVENT_REGION[1],
            "columns_inside_region":
                solvent_columns,
            "removed":
                solvent_columns == 0,
            "observed_gap":
                solvent_gap,
        },

        "artificial_color_region": {
            "lower_ppm":
                ARTIFICIAL_COLOR_REGION[0],
            "upper_ppm":
                ARTIFICIAL_COLOR_REGION[1],
            "spectral_points":
                color_region_columns,
        },

        "duplicates": {
            "exact_duplicate_spectra":
                duplicate_spectra,
            "duplicate_row_indexes":
                duplicate_rows,
        },

        "spectral_statistics": {
            "minimum":
                float(np.nanmin(array))
                if array.size
                else None,

            "maximum":
                float(np.nanmax(array))
                if array.size
                else None,

            "mean":
                float(np.nanmean(array))
                if array.size
                else None,

            "std":
                float(np.nanstd(array))
                if array.size
                else None,
        },

        "compatibility":
            compatibility,

        "overall_valid":
            overall_valid,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Audit report saved to: "
        f"{output_path}"
    )

    return report


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) >= 2:

        dataset_path = Path(
            sys.argv[1]
        )

        output_path = Path(
            f"{dataset_path.stem}_dataset_audit.json"
        )

    else:

        dataset_path = (
            DEFAULT_DATASET_PATH
        )

        output_path = (
            DEFAULT_OUTPUT_PATH
        )

    audit_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
    )