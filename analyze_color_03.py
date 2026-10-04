import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(
    "data/raw/color_03.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)
print()


# =========================================================
# Metadata
# =========================================================

sample_type = df.iloc[:, 0].astype(str).str.strip()
sample_id = df.iloc[:, 2].astype(str).str.strip()


print("Sample groups:")
print(sample_type.value_counts())
print()


# =========================================================
# Spectral columns
# =========================================================

spectral_columns = [
    column
    for column in df.columns
    if not column.startswith("Unnamed:")
]


X = df[spectral_columns].apply(
    pd.to_numeric,
    errors="coerce"
)


print("Spectral points:", X.shape[1])
print("Missing values:", X.isna().sum().sum())
print()


# =========================================================
# Basic spectral statistics
# =========================================================

print("=" * 70)
print("Basic spectral statistics")
print("=" * 70)

for group in sample_type.unique():

    group_X = X[sample_type == group]

    print()
    print("Group:", group)
    print("Samples:", len(group_X))

    mean_intensity = group_X.mean().mean()
    std_intensity = group_X.std().mean()

    print("Mean intensity:", round(mean_intensity, 6))
    print("Average point-wise std:", round(std_intensity, 6))


# =========================================================
# PCA
# =========================================================

print()
print("=" * 70)
print("PCA")
print("=" * 70)


# Fill missing values if any
X = X.fillna(X.median())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)


explained = pca.explained_variance_ratio_

print()
print("First 10 PCA explained variance:")

for i in range(min(10, len(explained))):
    print(
        f"PC{i + 1}: "
        f"{explained[i] * 100:.4f}%"
    )


print()
print(
    "First 2 PCs total:",
    round(
        (explained[0] + explained[1]) * 100,
        4
    ),
    "%"
)


# =========================================================
# PCA coordinates
# =========================================================

print()
print("=" * 70)
print("PCA coordinates")
print("=" * 70)

for i in range(len(df)):

    print(
        f"{sample_id.iloc[i]:>6} | "
        f"{sample_type.iloc[i]:<12} | "
        f"PC1={X_pca[i, 0]: .4f} | "
        f"PC2={X_pca[i, 1]: .4f}"
    )