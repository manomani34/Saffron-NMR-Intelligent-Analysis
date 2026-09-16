from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("data/raw/color.csv")

OUTPUT_CSV = Path("data/processed/color_labeled.csv")
OUTPUT_JSON = Path("data/processed/color_labels.json")


# ============================================================
# SAMPLE LABELS
# ============================================================

SAMPLE_LABELS = {
    1: {
        "sample_name": "sunset yellow(127)",
        "class": "pure_artificial_color",
        "subclass": "sunset_yellow_127",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    2: {
        "sample_name": "orange 2",
        "class": "pure_artificial_color",
        "subclass": "orange_2",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    3: {
        "sample_name": "Amuranth",
        "class": "pure_artificial_color",
        "subclass": "amaranth",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    4: {
        "sample_name": "Azorubin",
        "class": "pure_artificial_color",
        "subclass": "azorubin",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    5: {
        "sample_name": "Quinoline yellow",
        "class": "pure_artificial_color",
        "subclass": "quinoline_yellow",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    6: {
        "sample_name": "Ponceau 4R (96)",
        "class": "pure_artificial_color",
        "subclass": "ponceau_4r_96",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    7: {
        "sample_name": "Erythrosin",
        "class": "pure_artificial_color",
        "subclass": "erythrosin",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    8: {
        "sample_name": "Orange GG",
        "class": "pure_artificial_color",
        "subclass": "orange_gg",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    9: {
        "sample_name": "Tartarazin",
        "class": "pure_artificial_color",
        "subclass": "tartrazine",
        "artificial_color": True,
        "artificial_color_percent": 100.0,
        "saffron": False,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    10: {
        "sample_name": "saffron + sunset yellow(127)",
        "class": "saffron_plus_artificial_color",
        "subclass": "sunset_yellow_127_5_percent",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    11: {
        "sample_name": "saffron + orange GG",
        "class": "saffron_plus_artificial_color",
        "subclass": "orange_gg_5_percent",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    12: {
        "sample_name": "saffron + Quinoline yellow",
        "class": "saffron_plus_artificial_color",
        "subclass": "quinoline_yellow_5_percent",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    13: {
        "sample_name": "saffron + orange 2",
        "class": "saffron_plus_artificial_color",
        "subclass": "orange_2_5_percent",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    14: {
        "sample_name": "saffron + Ponceau 4R (96)",
        "class": "saffron_plus_artificial_color",
        "subclass": "ponceau_4r_96_5_percent",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    15: {
        "sample_name": "saffron + Erythrosin",
        "class": "saffron_plus_artificial_color",
        "subclass": "erythrosin_5_percent",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    16: {
        "sample_name": "saffron + orange 2",
        "class": "saffron_plus_artificial_color",
        "subclass": "orange_2_5_percent_replicate",
        "artificial_color": True,
        "artificial_color_percent": 5.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    17: {
        "sample_name": "saffron",
        "class": "real_saffron",
        "subclass": "real_saffron",
        "artificial_color": False,
        "artificial_color_percent": 0.0,
        "saffron": True,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": False,
    },

    18: {
        "sample_name": "650",
        "class": "unknown_adulterated_saffron",
        "subclass": "unknown_adulteration_650",
        "artificial_color": False,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": True,
        "known_artificial_color": False,
    },

    19: {
        "sample_name": "651",
        "class": "unknown_adulterated_saffron",
        "subclass": "unknown_adulteration_651",
        "artificial_color": False,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": True,
        "known_artificial_color": False,
    },

    20: {
        "sample_name": "saffron + Quinoline yellow 10 %",
        "class": "saffron_plus_artificial_color",
        "subclass": "quinoline_yellow_10_percent",
        "artificial_color": True,
        "artificial_color_percent": 10.0,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    21: {
        "sample_name": "saffron 2",
        "class": "real_saffron",
        "subclass": "real_saffron_2",
        "artificial_color": False,
        "artificial_color_percent": 0.0,
        "saffron": True,
        "adulterated": False,
        "unknown_adulteration": False,
        "known_artificial_color": False,
    },

    22: {
        "sample_name": "445",
        "class": "artificial_color_adulterated_saffron",
        "subclass": "azorubin_plus_orange_2",
        "artificial_color": True,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    23: {
        "sample_name": "372",
        "class": "artificial_color_adulterated_saffron",
        "subclass": "orange_2",
        "artificial_color": True,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    24: {
        "sample_name": "544",
        "class": "artificial_saffron",
        "subclass": "tartrazine",
        "artificial_color": True,
        "artificial_color_percent": None,
        "saffron": False,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    25: {
        "sample_name": "647",
        "class": "small_amount_artificial_color",
        "subclass": "small_amount_color_647",
        "artificial_color": True,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    26: {
        "sample_name": "648",
        "class": "small_amount_artificial_color",
        "subclass": "small_amount_color_648",
        "artificial_color": True,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },

    27: {
        "sample_name": "646",
        "class": "small_amount_artificial_color",
        "subclass": "small_amount_color_646",
        "artificial_color": True,
        "artificial_color_percent": None,
        "saffron": True,
        "adulterated": True,
        "unknown_adulteration": False,
        "known_artificial_color": True,
    },
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("  ", " ")
    )


def find_name_column(df):
    candidates = [
        "name",
        "name ",
        "name  ",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Sample name column was not found."
    )


# ============================================================
# LABEL DATASET
# ============================================================

def build_labels(df):
    name_column = find_name_column(df)

    rows = []

    for index, row in df.iterrows():

        sample_number = int(row["number"])

        if sample_number not in SAMPLE_LABELS:
            raise ValueError(
                f"No label defined for sample number "
                f"{sample_number}."
            )

        label = SAMPLE_LABELS[
            sample_number
        ]

        actual_name = str(
            row[name_column]
        ).strip()

        expected_name = (
            label["sample_name"]
        )

        rows.append(
            {
                "row_index": index,
                "number": sample_number,
                "name": actual_name,
                "expected_name": expected_name,
                "name_matches_definition":
                    normalize_name(actual_name)
                    == normalize_name(expected_name),
                "class": label["class"],
                "subclass": label["subclass"],
                "artificial_color":
                    label["artificial_color"],
                "artificial_color_percent":
                    label["artificial_color_percent"],
                "saffron":
                    label["saffron"],
                "adulterated":
                    label["adulterated"],
                "unknown_adulteration":
                    label["unknown_adulteration"],
                "known_artificial_color":
                    label["known_artificial_color"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("COLOR DATASET LABELING")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print()
    print(
        f"Input rows : {len(df)}"
    )

    labels = build_labels(df)

    print(
        f"Labeled rows : {len(labels)}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("[1] LABEL VALIDATION")

    mismatches = labels[
        ~labels["name_matches_definition"]
    ]

    print(
        f"Name mismatches : "
        f"{len(mismatches)}"
    )

    if not mismatches.empty:

        for _, row in mismatches.iterrows():
            print(
                f"  #{row['number']}: "
                f"actual='{row['name']}' | "
                f"expected='{row['expected_name']}'"
            )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("[2] CLASS DISTRIBUTION")

    class_counts = (
        labels["class"]
        .value_counts()
        .sort_index()
    )

    for class_name, count in class_counts.items():
        print(
            f"  {class_name:<40} "
            f"{count:>3}"
        )

    # --------------------------------------------------------
    # ARTIFICIAL COLOR DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("[3] ARTIFICIAL COLOR FLAG")

    color_counts = (
        labels["artificial_color"]
        .value_counts()
        .sort_index()
    )

    for value, count in color_counts.items():
        print(
            f"  artificial_color={value}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # ADULTERATION DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("[4] ADULTERATION FLAG")

    adulteration_counts = (
        labels["adulterated"]
        .value_counts()
        .sort_index()
    )

    for value, count in adulteration_counts.items():
        print(
            f"  adulterated={value}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # MERGE LABELS INTO ORIGINAL DATASET
    # --------------------------------------------------------

    labeled_df = df.merge(
        labels[
            [
                "number",
                "class",
                "subclass",
                "artificial_color",
                "artificial_color_percent",
                "saffron",
                "adulterated",
                "unknown_adulteration",
                "known_artificial_color",
            ]
        ],
        on="number",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labeled_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_CSV),
        "sample_count": len(labeled_df),
        "label_count": len(labels),
        "name_mismatches": len(mismatches),
        "class_distribution":
            class_counts.to_dict(),
        "artificial_color_distribution":
            {
                str(key): int(value)
                for key, value
                in color_counts.items()
            },
        "adulteration_distribution":
            {
                str(key): int(value)
                for key, value
                in adulteration_counts.items()
            },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"CSV saved  : {OUTPUT_CSV}"
    )

    print(
        f"JSON saved : {OUTPUT_JSON}"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()