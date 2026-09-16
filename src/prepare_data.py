import pandas as pd

from src.validate_data import SAMPLE_MAP


# --------------------------------------------------
# Confirmed duplicate measurements
# Chemistry team confirmed:
# 34 and 35 are the same measurement.
#
# We keep ID 34 and remove ID 35.
# --------------------------------------------------

CONFIRMED_DUPLICATE_IDS = {35}

METADATA_COLUMNS = [
    "name",
    "group",
    "region_code",
    "OriginalSampleName",
    "HarvestYear",
]


def extract_harvest_year(sample_name: str) -> int:
    """
    Extract harvest year from the original laboratory sample name.

    94-* -> 1394
    04-* -> 1404
    """

    if sample_name is None:
        raise ValueError(
            "Sample name cannot be None."
        )

    sample_name = str(sample_name).strip()

    if sample_name.startswith("94-"):
        return 1394

    if sample_name.startswith("04-"):
        return 1404

    raise ValueError(
        "Cannot determine harvest year from sample name: "
        f"{sample_name}"
    )


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    print("\n" + "=" * 70)
    print("PREPARING DATASET FOR MODELING")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Copy input data
    # --------------------------------------------------

    data = df.copy()

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------

    required_columns = [
        "name",
        "group",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError(
            "Input dataset is empty."
        )

    # --------------------------------------------------
    # 3. Normalize Sample ID
    # --------------------------------------------------

    print("\n[1] SAMPLE IDENTIFICATION")

    try:
        data["name"] = pd.to_numeric(
            data["name"],
            errors="raise",
        ).astype(int)

    except Exception as exc:
        raise ValueError(
            "Sample IDs in 'name' must be numeric."
        ) from exc

    data["OriginalSampleName"] = data["name"].map(
        SAMPLE_MAP
    )

    missing_original_names = (
        data["OriginalSampleName"].isna()
    )

    if missing_original_names.any():

        missing_ids = data.loc[
            missing_original_names,
            "name",
        ].tolist()

        raise ValueError(
            "No original sample name mapping found for "
            f"Sample IDs: {missing_ids}"
        )

    print(
        f"OK: Original sample names mapped for "
        f"{len(data)} records"
    )

    # --------------------------------------------------
    # 4. Extract harvest year
    # --------------------------------------------------

    print("\n[2] HARVEST YEAR EXTRACTION")

    data["HarvestYear"] = data[
        "OriginalSampleName"
    ].apply(
        extract_harvest_year
    )

    harvest_counts = (
        data["HarvestYear"]
        .value_counts()
        .sort_index()
    )

    for year, count in harvest_counts.items():
        print(
            f"{year} -> {count} records"
        )

    print(
        "OK: Harvest year extracted from confirmed "
        "sample-name convention"
    )

    # --------------------------------------------------
    # 5. Duplicate measurement handling
    # --------------------------------------------------

    print("\n[3] DUPLICATE MEASUREMENT HANDLING")

    original_count = len(data)

    duplicate_mask = data["name"].isin(
        CONFIRMED_DUPLICATE_IDS
    )

    removed_rows = data[
        duplicate_mask
    ].copy()

    data = data[
        ~duplicate_mask
    ].copy()

    print(
        f"Original records       : {original_count}"
    )

    print(
        f"Removed duplicate rows : {len(removed_rows)}"
    )

    print(
        f"Modeling records       : {len(data)}"
    )

    if not removed_rows.empty:

        removed_names = (
            removed_rows["OriginalSampleName"]
            .astype(str)
            .tolist()
        )

        removed_ids = (
            removed_rows["name"]
            .astype(int)
            .tolist()
        )

        print(
            f"Removed Sample IDs    : {removed_ids}"
        )

        print(
            f"Removed sample names  : {removed_names}"
        )

    # --------------------------------------------------
    # 6. Validate that data remains
    # --------------------------------------------------

    if data.empty:
        raise ValueError(
            "No records remain after duplicate removal."
        )

    # --------------------------------------------------
    # 7. Spectral columns
    # --------------------------------------------------

    spectral_columns = []

    for column in data.columns:

        if column in METADATA_COLUMNS:
            continue

        try:
            float(column)
            spectral_columns.append(column)

        except (ValueError, TypeError):
            continue

    if not spectral_columns:
        raise ValueError(
            "No valid numeric spectral columns were found."
        )

    print("\n[4] SPECTRAL DATA")

    print(
        f"Spectral points: "
        f"{len(spectral_columns)}"
    )

    # --------------------------------------------------
    # 8. X - spectral matrix
    # --------------------------------------------------

    X = data[
        spectral_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if X.isna().any().any():
        raise ValueError(
            "Spectral matrix contains missing values."
        )

    # --------------------------------------------------
    # 9. y - group labels
    # --------------------------------------------------

    y = data[
        "group"
    ].astype(str).copy()

    if y.nunique() < 2:
        raise ValueError(
            "At least two classes are required for modeling."
        )

    # --------------------------------------------------
    # 10. Metadata
    # --------------------------------------------------

    metadata = data[
        [
            "name",
            "OriginalSampleName",
            "group",
            "region_code",
            "HarvestYear",
        ]
    ].copy()

    metadata = metadata.rename(
        columns={
            "name": "SampleId",
            "group": "Group",
            "region_code": "RegionCode",
        }
    )

    # --------------------------------------------------
    # 11. Reset indexes consistently
    # --------------------------------------------------

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    # --------------------------------------------------
    # 12. Final modeling information
    # --------------------------------------------------

    print("\n[5] FINAL MODELING DATA")

    print(
        f"X shape        : {X.shape}"
    )

    print(
        f"y shape        : {y.shape}"
    )

    print(
        f"Metadata shape : {metadata.shape}"
    )

    # --------------------------------------------------
    # 13. Group distribution
    # --------------------------------------------------

    print("\nGroup distribution:")

    group_counts = (
        y.value_counts()
        .sort_index(
            key=lambda values: values.map(
                lambda x:
                    int(str(x)[1:])
                    if str(x).startswith("G")
                    else 999
            )
        )
    )

    for group, count in group_counts.items():
        print(
            f"{group:4} -> {count} samples"
        )

    # --------------------------------------------------
    # 14. Harvest-year distribution
    # --------------------------------------------------

    print("\nHarvest year distribution:")

    final_harvest_counts = (
        metadata["HarvestYear"]
        .value_counts()
        .sort_index()
    )

    for year, count in final_harvest_counts.items():
        print(
            f"{year} -> {count} samples"
        )

    # --------------------------------------------------
    # 15. Verify Sample IDs
    # --------------------------------------------------

    print("\n[6] SAMPLE ID CHECK")

    sample_ids = metadata[
        "SampleId"
    ].tolist()

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError(
            "Duplicate Sample IDs remain after "
            "duplicate measurement removal."
        )

    print(
        f"Unique Sample IDs : "
        f"{len(sample_ids)}"
    )

    print(
        "OK: Sample IDs are unique in modeling dataset"
    )

    # --------------------------------------------------
    # 16. Verify metadata consistency
    # --------------------------------------------------

    print("\n[7] METADATA CHECK")

    if metadata[
        "OriginalSampleName"
    ].isna().any():

        raise ValueError(
            "Some OriginalSampleName values are missing."
        )

    if metadata[
        "HarvestYear"
    ].isna().any():

        raise ValueError(
            "Some HarvestYear values are missing."
        )

    print(
        "OK: Original sample names are available"
    )

    print(
        "OK: Harvest years are available"
    )

    # --------------------------------------------------
    # 17. Verify dimensions
    # --------------------------------------------------

    print("\n[8] DIMENSION CHECK")

    if not (
        len(X)
        == len(y)
        == len(metadata)
    ):
        raise ValueError(
            "X, y and metadata row counts do not match."
        )

    print(
        "OK: X, y and metadata have matching row counts"
    )

    # --------------------------------------------------
    # 18. Final
    # --------------------------------------------------

    print("\nOK: Dataset is ready for analysis")

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETED")
    print("=" * 70)

    print(
        f"Final X shape        : {X.shape}"
    )

    print(
        f"Final y shape        : {y.shape}"
    )

    print(
        f"Final metadata shape : {metadata.shape}"
    )

    return X, y, metadata