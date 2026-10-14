from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
REGISTRY_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "sample_registry.csv"

df = pd.read_csv(DATA_FILE, keep_default_na=False)
registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)

metadata = ["#", "Unnamed: 1", "Title", "Class"]
ppm_cols = [
    c for c in df.columns
    if c not in metadata and pd.notna(pd.to_numeric(c, errors="coerce"))
]
X = df[ppm_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

# بررسی شباهت شکل طیف‌ها پس از حذف اثر مقیاس کلی
norms = np.linalg.norm(X, axis=1)
X_norm = X / np.maximum(norms[:, None], 1e-12)
similarity = X_norm @ X_norm.T

groups = [
    ("saffron_reference_candidate", "Saffron candidates"),
    ("pure_color_reference_candidate", "Pure-color candidates"),
    ("adulterated_reference_candidate", "Saffron+color candidates"),
]

for group, label in groups:
    idx = np.where(registry["initial_group"].to_numpy() == group)[0]
    if len(idx) == 0:
        continue

    values = X[idx]
    pairwise = similarity[np.ix_(idx, idx)]
    upper = pairwise[np.triu_indices(len(idx), k=1)]

    print(f"\n--- {label} (n={len(idx)}) ---")
    print(f"Median spectrum intensity: {np.median(values):.5f}")
    print(f"Median sample maximum: {np.median(values.max(axis=1)):.5f}")
    print(f"Median sample standard deviation: {np.median(values.std(axis=1)):.5f}")
    if len(upper):
        print(f"Median within-group cosine similarity: {np.median(upper):.5f}")

print("\nNote: group labels are preliminary, not confirmed ground truth.")
