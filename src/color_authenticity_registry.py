from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data/raw/total-color-data.csv"
OUTPUT_PATH = BASE_DIR / "reports/color_authenticity_new/sample_registry.csv"


def classify_sample(row):
    title = str(row.get("Title", "")).strip().lower()
    description = str(row.get("Unnamed: 1", "")).strip().lower()
    sample_class = str(row.get("Class", "")).strip().lower()

    text = " ".join([title, description, sample_class])

    if title == "test":
        return "test_sample", "unverified"

    if title == "saffron":
        return "saffron_reference_candidate", "needs_reference_confirmation"

    if title == "saffron+color":
        return "adulterated_reference_candidate", "needs_metadata_confirmation"

    if "saffron" in text and any(
        color in text
        for color in [
            "sunset", "orange", "amaranth", "azorubin",
            "quinoline", "ponceau", "erythros", "tartrazine",
            "127", "96",
        ]
    ):
        return "saffron_color_mixture_candidate", "needs_metadata_confirmation"

    if any(
        color in sample_class
        for color in [
            "sunset yellow", "orange 2", "amaranth", "azorubine",
            "quinoline yellow", "ponceau", "erythrozine",
            "erythrosine", "orange ggn", "tartrazine",
        ]
    ):
        return "pure_color_reference_candidate", "needs_reference_confirmation"

    return "unclassified", "unverified"


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    registry = pd.DataFrame({
        "source_row": range(1, len(df) + 1),
        "sample_number": df["#"],
        "title": df["Title"],
        "description": df["Unnamed: 1"],
        "class_value": df["Class"],
    })

    classifications = df.apply(classify_sample, axis=1)
    registry["initial_group"] = [item[0] for item in classifications]
    registry["verification_status"] = [item[1] for item in classifications]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Registry created: {OUTPUT_PATH}")
    print("\nInitial groups:")
    print(registry["initial_group"].value_counts().to_string())
    print("\nVerification statuses:")
    print(registry["verification_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
