from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports" / "color_authenticity_new"
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"

registry = pd.read_csv(
    REPORT_DIR / "sample_registry.csv",
    keep_default_na=False,
)
distance_file = REPORT_DIR / "distance_to_saffron_reference.csv"
distances = pd.read_csv(distance_file)

df = pd.read_csv(DATA_FILE, keep_default_na=False)

metadata_cols = ["#", "Unnamed: 1", "Title", "Class"]
ppm_cols = [
    c for c in df.columns
    if c not in metadata_cols
    and pd.notna(pd.to_numeric(c, errors="coerce"))
]

X = df[ppm_cols].apply(pd.to_numeric, errors="raise").to_numpy(float)

# SNV: standardize each spectrum independently.
means = X.mean(axis=1, keepdims=True)
stds = X.std(axis=1, keepdims=True)
X_snv = (X - means) / np.maximum(stds, 1e-12)

reference_mask = registry["initial_group"].eq(
    "saffron_reference_candidate"
).to_numpy()

reference_spectra = X_snv[reference_mask]
reference_mean = reference_spectra.mean(axis=0, keepdims=True)

# Cosine similarity against the mean saffron reference spectrum.
similarities = cosine_similarity(X_snv, reference_mean).ravel()

result = registry[
    [
        "source_row",
        "sample_number",
        "class_value",
        "title",
        "initial_group",
        "verification_status",
    ]
].copy()

result["cosine_similarity_to_saffron"] = similarities

distance_col = next(
    c for c in distances.columns if "distance" in c.lower()
)

distance_keys = [
    c for c in ["source_row", "sample_number", "class_value"]
    if c in distances.columns and c in result.columns
]

if "source_row" in distance_keys:
    result = result.merge(
        distances[["source_row", distance_col]].drop_duplicates("source_row"),
        on="source_row",
        how="left",
    )
elif "sample_number" in distance_keys:
    result = result.merge(
        distances[["sample_number", distance_col]].drop_duplicates("sample_number"),
        on="sample_number",
        how="left",
    )
else:
    raise ValueError("No safe key found to join distance results.")

result = result.rename(
    columns={distance_col: "distance_to_saffron_reference"}
)

# This is a comparison label only, not an authenticity verdict.
result["metric_agreement"] = np.select(
    [
        (result["distance_to_saffron_reference"] <= 19.996)
        & (result["cosine_similarity_to_saffron"] >= 0.90),
        (result["distance_to_saffron_reference"] > 19.996)
        & (result["cosine_similarity_to_saffron"] < 0.90),
    ],
    [
        "both_metrics_close",
        "both_metrics_different",
    ],
    default="metrics_disagree_or_borderline",
)

output = REPORT_DIR / "distance_cosine_comparison.csv"
result.to_csv(output, index=False, encoding="utf-8-sig")

print(f"Samples analyzed: {len(result)}")
print("\nMetric comparison counts:")
print(result["metric_agreement"].value_counts().to_string())

print("\nSamples requiring closer review:")
print(
    result[
        result["metric_agreement"] != "both_metrics_close"
    ][
        [
            "sample_number",
            "class_value",
            "initial_group",
            "distance_to_saffron_reference",
            "cosine_similarity_to_saffron",
            "metric_agreement",
        ]
    ]
    .sort_values("distance_to_saffron_reference")
    .round(4)
    .to_string(index=False)
)

print(f"\nSaved: {output}")
