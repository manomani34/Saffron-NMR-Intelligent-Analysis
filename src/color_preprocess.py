from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/processed/color_labeled.csv"
)

OUTPUT_X = Path(
    "data/processed/color_X.npy"
)

OUTPUT_PPM = Path(
    "data/processed/color_ppm.npy"
)

OUTPUT_META = Path(
    "data/processed/color_preprocessed.csv"
)

OUTPUT_REPORT = Path(
    "data/processed/color_preprocess_report.json"
)

SOLVENT_REGION = (
    4.66797,
    5.49960,
)

COLOR_REGION = (
    5.0,
    9.0,
)


# ============================================================
# HELPERS
# ============================================================

def is_numeric_column(column):
    try:
        float(str(column).strip())
        return True
    except (ValueError, TypeError):
        return False


def get_spectral_columns(df):
    return [
        column
        for column in df.columns
        if is_numeric_column(column)
    ]


def get_ppm_array(columns):
    return np.asarray(
        [
            float(str(column).strip())
            for column in columns
        ],
        dtype=float,
    )


def find_name_column(df):
    for column in df.columns:

        if str(column).strip().lower() == "name":
            return column

    raise ValueError(
        "Sample name column was not found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR DATASET PREPROCESSING")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    print()
    print(
        f"Input rows : {len(df)}"
    )

    # --------------------------------------------------------
    # NAME COLUMN
    # --------------------------------------------------------

    name_column = find_name_column(df)

    print(
        f"Sample name column : "
        f"{name_column!r}"
    )

    # --------------------------------------------------------
    # SPECTRAL COLUMNS
    # --------------------------------------------------------

    spectral_columns = get_spectral_columns(df)

    if not spectral_columns:
        raise ValueError(
            "No spectral columns found."
        )

    ppm = get_ppm_array(
        spectral_columns
    )

    print(
        f"Original spectral points : "
        f"{len(spectral_columns)}"
    )

    print(
        f"PPM range : "
        f"{ppm.min():.6f} - "
        f"{ppm.max():.6f}"
    )

    # --------------------------------------------------------
    # SOLVENT MASK
    # --------------------------------------------------------

    solvent_mask = (
        (ppm >= SOLVENT_REGION[0])
        & (ppm <= SOLVENT_REGION[1])
    )

    keep_mask = ~solvent_mask

    removed_count = int(
        solvent_mask.sum()
    )

    kept_count = int(
        keep_mask.sum()
    )

    ppm_kept = ppm[
        keep_mask
    ]

    print()
    print("[1] SOLVENT REMOVAL")

    print(
        f"Solvent region : "
        f"{SOLVENT_REGION[0]} - "
        f"{SOLVENT_REGION[1]} ppm"
    )

    print(
        f"Removed points : "
        f"{removed_count}"
    )

    print(
        f"Remaining points : "
        f"{kept_count}"
    )

    # --------------------------------------------------------
    # BUILD X
    # --------------------------------------------------------

    X_full = (
        df[spectral_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(X_full).all():
        raise ValueError(
            "Spectral matrix contains "
            "non-finite values."
        )

    X = X_full[
        :,
        keep_mask,
    ]

    print()
    print("[2] MATRIX")

    print(
        f"X shape : {X.shape}"
    )

    # --------------------------------------------------------
    # ARTIFICIAL COLOR REGION
    # --------------------------------------------------------

    color_mask = (
        (ppm_kept >= COLOR_REGION[0])
        & (ppm_kept <= COLOR_REGION[1])
    )

    color_points = int(
        color_mask.sum()
    )

    print()
    print("[3] ARTIFICIAL COLOR REGION")

    print(
        f"Region : "
        f"{COLOR_REGION[0]} - "
        f"{COLOR_REGION[1]} ppm"
    )

    print(
        f"Points : "
        f"{color_points}"
    )

    # --------------------------------------------------------
    # REQUIRED METADATA
    # --------------------------------------------------------

    required_metadata = [
        "number",
        "class",
        "subclass",
        "artificial_color",
        "artificial_color_percent",
        "saffron",
        "adulterated",
        "unknown_adulteration",
        "known_artificial_color",
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in df.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Missing metadata columns: "
            f"{missing_metadata}"
        )

    # --------------------------------------------------------
    # BUILD METADATA
    # --------------------------------------------------------

    meta = pd.DataFrame(
        {
            "row_index":
                np.arange(len(df)),

            "number":
                df["number"],

            "name":
                df[name_column]
                .astype(str)
                .str.strip(),

            "class":
                df["class"],

            "subclass":
                df["subclass"],

            "artificial_color":
                df["artificial_color"],

            "artificial_color_percent":
                df["artificial_color_percent"],

            "saffron":
                df["saffron"],

            "adulterated":
                df["adulterated"],

            "unknown_adulteration":
                df["unknown_adulteration"],

            "known_artificial_color":
                df["known_artificial_color"],
        }
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_X.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_X,
        X,
    )

    np.save(
        OUTPUT_PPM,
        ppm_kept,
    )

    meta.to_csv(
        OUTPUT_META,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "input": str(INPUT_CSV),

        "output": {
            "X":
                str(OUTPUT_X),

            "ppm":
                str(OUTPUT_PPM),

            "metadata":
                str(OUTPUT_META),
        },

        "samples":
            int(X.shape[0]),

        "spectral_points": {
            "original":
                int(len(ppm)),

            "removed_solvent":
                removed_count,

            "remaining":
                int(len(ppm_kept)),
        },

        "ppm_range": {
            "original_min":
                float(ppm.min()),

            "original_max":
                float(ppm.max()),

            "processed_min":
                float(ppm_kept.min()),

            "processed_max":
                float(ppm_kept.max()),
        },

        "solvent_region": {
            "lower_ppm":
                SOLVENT_REGION[0],

            "upper_ppm":
                SOLVENT_REGION[1],

            "removed_points":
                removed_count,
        },

        "artificial_color_region": {
            "lower_ppm":
                COLOR_REGION[0],

            "upper_ppm":
                COLOR_REGION[1],

            "points":
                color_points,
        },

        "X_shape": [
            int(X.shape[0]),
            int(X.shape[1]),
        ],
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print()
    print(
        f"X        : {OUTPUT_X}"
    )

    print(
        f"PPM      : {OUTPUT_PPM}"
    )

    print(
        f"Metadata : {OUTPUT_META}"
    )

    print(
        f"Report   : {OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()