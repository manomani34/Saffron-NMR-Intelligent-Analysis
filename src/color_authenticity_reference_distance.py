from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
REGISTRY_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "sample_registry.csv"
OUTPUT_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "distance_to_saffron_reference.csv"

df = pd.read_csv(DATA_FILE, keep_default_na=False)
registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)

metadata = ["#", "Unnamed: 1", "Title", "Class"]
ppm_cols = [
    c for c in df.columns
    if c not in metadata and pd.notna(pd.to_numeric(c, errors="coerce"))
]
X = df[ppm_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

# SNV normalization for each spectrum
means = X.mean(axis=1, keepdims=True)
stds = X.std(axis=1, keepdims=True)
X_snv = (X - means) / np.maximum(stds, 1e-12)

reference_mask = (
    registry["initial_group"] == "saffron_reference_candidate"
).to_numpy()

if reference_mask.sum() < 2:
    raise ValueError("Not enough saffron reference candidates.")

reference = X_snv[reference_mask].mean(axis=0)

# Euclidean distance to the candidate saffron reference
distances = np.linalg.norm(X_snv - reference, axis=1)

result = registry[
    ["source_row", "sample_number", "title", "description", "class_value", "initial_group"]
].copy()
result["distance_to_saffron_reference"] = distances

result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Median distance by preliminary group:")
print(
    result.groupby("initial_group")["distance_to_saffron_reference"]
    .agg(["count", "median", "min", "max"])
    .round(3)
    .to_string()
)

print("\nClosest samples to saffron reference:")
print(
    result.sort_values("distance_to_saffron_reference")
    [["sample_number", "class_value", "initial_group", "distance_to_saffron_reference"]]
    .head(10)
    .round(3)
    .to_string(index=False)
)

print(f"\nSaved: {OUTPUT_FILE}")
