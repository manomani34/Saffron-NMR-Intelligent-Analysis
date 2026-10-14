from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "color_authenticity_new"

df = pd.read_csv(DATA_FILE, keep_default_na=False)

metadata_columns = ["#", "Unnamed: 1", "Title", "Class"]
ppm_columns = [
    col for col in df.columns
    if col not in metadata_columns and pd.to_numeric(
        pd.Series([col]), errors="coerce"
    ).notna().all()
]

X = df[ppm_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

summary = pd.DataFrame({
    "source_row": np.arange(1, len(df) + 1),
    "sample_number": df["#"],
    "title": df["Title"],
    "class_value": df["Class"],
    "minimum_intensity": np.min(X, axis=1),
    "maximum_intensity": np.max(X, axis=1),
    "mean_intensity": np.mean(X, axis=1),
    "std_intensity": np.std(X, axis=1),
    "negative_value_count": np.sum(X < 0, axis=1),
    "zero_value_count": np.sum(X == 0, axis=1),
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / "spectral_quality_report.csv"
summary.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Samples: {X.shape[0]}")
print(f"Spectral features: {X.shape[1]}")
print(f"PPM range: {min(map(float, ppm_columns)):.5f} to {max(map(float, ppm_columns)):.5f}")
print(f"All intensities finite: {np.isfinite(X).all()}")
print(f"Report: {output_file}")
print()
print(summary[[
    "minimum_intensity",
    "maximum_intensity",
    "mean_intensity",
    "std_intensity",
    "negative_value_count",
    "zero_value_count",
]].describe().round(4).to_string())
