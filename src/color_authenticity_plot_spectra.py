from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "total-color-data.csv"
REGISTRY_FILE = BASE_DIR / "reports" / "color_authenticity_new" / "sample_registry.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "color_authenticity_new"

df = pd.read_csv(DATA_FILE, keep_default_na=False)
registry = pd.read_csv(REGISTRY_FILE, keep_default_na=False)

metadata_columns = ["#", "Unnamed: 1", "Title", "Class"]
ppm_columns = [
    col for col in df.columns
    if col not in metadata_columns and pd.notna(pd.to_numeric(col, errors="coerce"))
]
ppm = pd.to_numeric(pd.Index(ppm_columns), errors="raise").to_numpy()
X = df[ppm_columns].apply(pd.to_numeric, errors="raise").to_numpy()

groups = [
    ("saffron_reference_candidate", "Saffron reference candidates"),
    ("pure_color_reference_candidate", "Pure-color reference candidates"),
    ("adulterated_reference_candidate", "Saffron + color candidates"),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax, (group_name, title) in zip(axes, groups):
    indices = registry.index[
        registry["initial_group"] == group_name
    ].tolist()

    for i in indices:
        ax.plot(ppm, X[i], linewidth=0.8, alpha=0.65)

    ax.set_title(f"{title} (n={len(indices)})")
    ax.set_ylabel("Raw intensity")
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Chemical shift (ppm)")
fig.suptitle("New color dataset — raw spectral overview", fontsize=14)
fig.tight_layout()

output_file = OUTPUT_DIR / "spectral_overview.png"
fig.savefig(output_file, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_file}")
