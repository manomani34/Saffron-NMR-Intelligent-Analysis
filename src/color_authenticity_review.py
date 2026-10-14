from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REGISTRY = BASE / "reports/color_authenticity_new/sample_registry.csv"
OUTPUT = BASE / "reports/color_authenticity_new/sample_review.csv"

df = pd.read_csv(REGISTRY, keep_default_na=False)

df["review_decision"] = "pending"
df["confirmed_authenticity"] = "unknown"
df["confirmed_color_type"] = "unknown"
df["confirmed_for_reference"] = False
df["review_notes"] = ""

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Created: {OUTPUT}")
print(f"Rows: {len(df)}")
print("All labels are pending; no sample was automatically approved as a reference.")
