import pandas as pd
from pathlib import Path


REGION_MAPPING = {
    "G1": "SEMNAN",
    "G2": "ISFAHAN",
    "G3": "ZANJAN",
    "G4": "ARAK",
    "G5": "NORTH_KHORASAN",
    "G6": "RAZAVI_KHORASAN",
    "G7": "FARS",
    "G8": "QOM",
    "G9": "YAZD",
    "G10": "QAZVIN",
    "G11": "LORESTAN",
}


def load_dataset(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8",
    )

    if "group" not in df.columns:
        raise ValueError(
            "Required column 'group' not found in dataset."
        )

    # --------------------------------------------------
    # Create region code
    # --------------------------------------------------

    region_code = df["group"].map(
        REGION_MAPPING
    )

    unknown_groups = df.loc[
        region_code.isna(),
        "group",
    ].unique()

    if len(unknown_groups) > 0:
        raise ValueError(
            f"Unknown group codes found: "
            f"{list(unknown_groups)}"
        )

    # Add metadata column in one operation
    df = pd.concat(
        [
            df,
            region_code.rename("region_code"),
        ],
        axis=1,
    )

    return df