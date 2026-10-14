from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
REGISTRY_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "sample_registry.csv"
OUTPUT_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "snv_quality_report.csv"

df = pd.read_csv(DATA_FILE, keep_default_na=False)
registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)

metadata = ["#", "Unnamed: 1", "Title", "Class"]
ppm_cols = [
    c for c in df.columns
    if c not in metadata and pd.notna(pd.to_numeric(c, errors="coerce"))
]
X = df[ppm_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

# SNV: centering and scaling each spectrum independently
row_means = X.mean(axis=1, keepdims=True)
row_stds = X.std(axis=1, keepdims=True)
X_snv = (X - row_means) / np.maximum(row_stds, 1e-12)

rows = []
for i, row in registry.iterrows():
    rows.append({
        "sample_number": row["sample_number"],
        "class_value": row["class_value"],
        "initial_group": row["initial_group"],
        "raw_mean": float(X[i].mean()),
        "raw_std": float(X[i].std()),
        "snv_mean": float(X_snv[i].mean()),
        "snv_std": float(X_snv[i].std()),
        "snv_min": float(X_snv[i].min()),
        "snv_max": float(X_snv[i].max()),
    })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("SNV quality check")
print(f"Samples: {len(X_snv)}")
print(f"Features per sample: {X_snv.shape[1]}")
print(f"All SNV values finite: {np.isfinite(X_snv).all()}")
print(f"Median SNV mean: {np.median(X_snv.mean(axis=1)):.8f}")
print(f"Median SNV standard deviation: {np.median(X_snv.std(axis=1)):.8f}")
print(f"Report: {OUTPUT_FILE}")
print()
print(result.groupby("initial_group")[["snv_min", "snv_max"]].median().round(4).to_string())
