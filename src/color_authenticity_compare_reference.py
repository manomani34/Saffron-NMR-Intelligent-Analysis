from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports" / "color_authenticity_new"

loo = pd.read_csv(REPORT_DIR / "saffron_reference_loo.csv")
distances = pd.read_csv(REPORT_DIR / "distance_to_saffron_reference.csv")
registry = pd.read_csv(REPORT_DIR / "sample_registry.csv")

print("=== LOO REFERENCE RANGE ===")
print(f"95th percentile: {loo['loo_distance'].quantile(0.95):.3f}")
print(f"Maximum: {loo['loo_distance'].max():.3f}")

print("\n=== SAMPLE DISTANCES ===")
columns = [
    c for c in [
        "sample_number",
        "class_value",
        "initial_group",
        "distance_to_saffron_reference",
    ]
    if c in distances.columns
]

if "initial_group" not in distances.columns:
    distances = distances.merge(
        registry[["sample_number", "initial_group"]],
        on="sample_number",
        how="left",
    )
    columns = [
        c for c in [
            "sample_number",
            "class_value",
            "initial_group",
            "distance_to_saffron_reference",
        ]
        if c in distances.columns
    ]

distance_col = next(
    c for c in distances.columns
    if "distance" in c.lower()
)

print(
    distances.sort_values(distance_col, ascending=True)[columns]
    .round(3)
    .to_string(index=False)
)
