from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
REGISTRY_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "sample_registry.csv"
OUTPUT_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "saffron_reference_loo.csv"

df = pd.read_csv(DATA_FILE, keep_default_na=False)
registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)

metadata = ["#", "Unnamed: 1", "Title", "Class"]
ppm_cols = [
    c for c in df.columns
    if c not in metadata and pd.notna(pd.to_numeric(c, errors="coerce"))
]
X = df[ppm_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

# Normalize each spectrum independently
X_snv = (X - X.mean(axis=1, keepdims=True)) / np.maximum(
    X.std(axis=1, keepdims=True), 1e-12
)

mask = registry["initial_group"].eq("saffron_reference_candidate").to_numpy()
indices = np.flatnonzero(mask)
X_ref = X_snv[indices]

loo_distances = []
for i in range(len(indices)):
    train_reference = np.delete(X_ref, i, axis=0).mean(axis=0)
    distance = np.linalg.norm(X_ref[i] - train_reference)
    loo_distances.append(distance)

result = registry.loc[
    indices,
    ["source_row", "sample_number", "class_value", "initial_group"]
].copy()
result["loo_distance"] = loo_distances

result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

values = np.asarray(loo_distances)
print(f"Reference candidates: {len(values)}")
print(f"LOO median distance: {np.median(values):.3f}")
print(f"LOO mean distance: {np.mean(values):.3f}")
print(f"LOO standard deviation: {np.std(values, ddof=1):.3f}")
print(f"LOO minimum: {np.min(values):.3f}")
print(f"LOO maximum: {np.max(values):.3f}")
print(f"LOO 95th percentile: {np.quantile(values, 0.95):.3f}")
print("\nLargest LOO distances:")
print(
    result.sort_values("loo_distance", ascending=False)
    [["sample_number", "class_value", "loo_distance"]]
    .head(5).round(3).to_string(index=False)
)
print(f"\nSaved: {OUTPUT_FILE}")
