import pandas as pd
import numpy as np


# Load files
color = pd.read_csv("data/raw/color.csv", low_memory=False)
color_02 = pd.read_csv("data/raw/color_02.csv", low_memory=False)


# Samples that exist in both files
sample_names = [
    "650",
    "651",
    "445",
    "372",
    "544",
    "647",
    "648",
    "646",
]


# Find common spectral columns
common_columns = [
    column
    for column in color.columns[3:]
    if column in color_02.columns[3:]
    and not column.startswith("Unnamed:")
]


print("Common spectral points:", len(common_columns))

if common_columns:
    print("Range:", common_columns[0], "to", common_columns[-1])

print()


for name in sample_names:

    row_a = color[
        color.iloc[:, 0].astype(str).str.strip() == name
    ]

    row_b = color_02[
        color_02.iloc[:, 0].astype(str).str.strip() == name
    ]

    print(f"Sample: {name}")

    if len(row_a) == 0:
        print("  color.csv: NOT FOUND")

    if len(row_b) == 0:
        print("  color_02.csv: NOT FOUND")

    if len(row_a) == 0 or len(row_b) == 0:
        print()
        continue

    x = row_a[common_columns].iloc[0].to_numpy(dtype=float)
    y = row_b[common_columns].iloc[0].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        print("  Not enough valid spectral points")
        print()
        continue

    correlation = np.corrcoef(x, y)[0, 1]
    mae = np.mean(np.abs(x - y))

    print("  Valid points:", len(x))
    print("  Correlation:", round(correlation, 6))
    print("  MAE:", round(mae, 6))
    print()