from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
REGISTRY_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "sample_registry.csv"
OUTPUT_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "pure_color_similarity.csv"

df = pd.read_csv(DATA_FILE, keep_default_na=False)
registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)

metadata = ["#", "Unnamed: 1", "Title", "Class"]
ppm_cols = [
    c for c in df.columns
    if c not in metadata and pd.notna(pd.to_numeric(c, errors="coerce"))
]
X = df[ppm_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

mask = registry["initial_group"].eq("pure_color_reference_candidate").to_numpy()
indices = np.flatnonzero(mask)

X_color = X[indices]
norms = np.linalg.norm(X_color, axis=1, keepdims=True)
X_norm = X_color / np.maximum(norms, 1e-12)
similarity = X_norm @ X_norm.T

rows = []
for local_i, global_i in enumerate(indices):
    others = np.delete(similarity[local_i], local_i)
    rows.append({
        "sample_number": registry.iloc[global_i]["sample_number"],
        "class_value": registry.iloc[global_i]["class_value"],
        "description": registry.iloc[global_i]["description"],
        "mean_similarity_to_other_colors": float(np.mean(others)),
        "median_similarity_to_other_colors": float(np.median(others)),
        "minimum_similarity_to_other_colors": float(np.min(others)),
    })

result = pd.DataFrame(rows).sort_values(
    "mean_similarity_to_other_colors"
)
result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(result.round(4).to_string(index=False))
print(f"\nSaved: {OUTPUT_FILE}")
