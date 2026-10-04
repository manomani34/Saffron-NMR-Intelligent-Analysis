import pandas as pd
import numpy as np


color_02 = pd.read_csv("data/raw/color_02.csv", low_memory=False)
color_03 = pd.read_csv("data/raw/color_03.csv", low_memory=False)


# ---------------------------------------------------------
# Common spectral columns
# ---------------------------------------------------------

spectral_02 = [
    c for c in color_02.columns
    if not c.startswith("Unnamed:")
]

spectral_03 = [
    c for c in color_03.columns
    if not c.startswith("Unnamed:")
]

common = [
    c for c in spectral_03
    if c in spectral_02
]


print("color_02 shape:", color_02.shape)
print("color_03 shape:", color_03.shape)
print()

print("Common spectral points:", len(common))

if common:
    print("Range:", common[0], "to", common[-1])

print()
print("=" * 70)
print()


# ---------------------------------------------------------
# Samples to compare
# ---------------------------------------------------------

samples = [
    ("04_4", "Saffron 04-4"),
    ("04_8", "Saffron 04-8"),
    ("04_10", "saffron 04-10"),
    ("04_16", "Saffron 04-16"),
    ("900", "900"),
    ("901", "901"),
    ("902", "902"),
]


for sample_id, name_02 in samples:

    # Find in color_03
    row_03 = color_03[
        color_03.iloc[:, 2].astype(str).str.strip() == sample_id
    ]

    # Find in color_02
    row_02 = color_02[
        color_02.iloc[:, 0].astype(str).str.strip().str.lower()
        == name_02.lower()
    ]

    print(f"Sample: {sample_id}")
    print(f"  color_02 name: {name_02}")

    if len(row_02) == 0:
        print("  color_02: NOT FOUND")

    if len(row_03) == 0:
        print("  color_03: NOT FOUND")

    if len(row_02) == 0 or len(row_03) == 0:
        print()
        continue

    x = row_02[common].iloc[0].to_numpy(dtype=float)
    y = row_03[common].iloc[0].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    print("  Valid points:", len(x))

    if len(x) >= 2:
        correlation = np.corrcoef(x, y)[0, 1]
        mae = np.mean(np.abs(x - y))

        print("  Correlation:", round(correlation, 6))
        print("  MAE:", round(mae, 6))

    print()