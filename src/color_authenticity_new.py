from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "raw" / "total-color-data.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "color_authenticity_new"

METADATA_COLUMNS = ["#", "Unnamed: 1", "Title", "Class"]


def is_ppm_column(column):
    try:
        float(str(column).strip())
        return True
    except (ValueError, TypeError):
        return False


def load_and_validate_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    missing_columns = [
        column for column in METADATA_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")

    ppm_columns = [
        column for column in df.columns
        if is_ppm_column(column)
    ]
    if not ppm_columns:
        raise ValueError("No spectral columns were found.")

    X = df[ppm_columns].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError("Spectral data contains missing or non-numeric values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Spectral data contains infinite values.")

    metadata = df[METADATA_COLUMNS].copy()

    return metadata, X, ppm_columns


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata, X, ppm_columns = load_and_validate_data()

    metadata.to_csv(
        OUTPUT_DIR / "sample_metadata.csv",
        index=False,
    )
    X.to_csv(
        OUTPUT_DIR / "spectral_data.csv",
        index=False,
    )

    report = {
        "input_file": str(INPUT_PATH),
        "sample_count": int(len(metadata)),
        "spectral_feature_count": int(len(ppm_columns)),
        "first_ppm": float(ppm_columns[0]),
        "last_ppm": float(ppm_columns[-1]),
        "missing_spectral_values": int(X.isna().sum().sum()),
        "infinite_spectral_values": int(
            (~np.isfinite(X.to_numpy(dtype=float))).sum()
        ),
        "title_distribution": {
            str(key): int(value)
            for key, value in metadata["Title"].fillna("(missing)")
            .value_counts().items()
        },
        "note": (
            "No authenticity labels or model predictions were assigned. "
            "Sample identities and reference status require verification."
        ),
    }

    pd.Series(report).to_json(
        OUTPUT_DIR / "validation_report.json",
        force_ascii=False,
        indent=2,
    )

    print("Validation completed successfully.")
    print(f"Samples: {len(metadata)}")
    print(f"Spectral features: {len(ppm_columns)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
