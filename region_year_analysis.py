from pathlib import Path
import warnings
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    pairwise_distances,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "saffron.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "region_year_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

TOP_REGIONS = 3

TOP_K_FEATURES = [
    10,
    20,
    30,
    50,
]

N_SPLITS = 3
N_REPEATS = 10

RANDOM_STATE = 42

PERMUTATIONS = 2000


# ============================================================
# REGION LABELS
# ============================================================

REGION_LABELS = {
    "G1": "سمنان",
    "G2": "اصفهان",
    "G3": "زنجان",
    "G4": "اراک",
    "G5": "خراسان شمالی",
    "G6": "خراسان رضوی",
    "G7": "فارس",
    "G8": "قم",
    "G9": "یزد",
    "G10": "قزوین",
    "G11": "لرستان",
}


# ============================================================
# SAMPLE METADATA
# ============================================================

SAMPLE_METADATA = {

    # G1 — سمنان
    1: {
        "Region": "G1",
        "RegionLabel": "سمنان",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-1",
    },
    2: {
        "Region": "G1",
        "RegionLabel": "سمنان",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-2",
    },
    3: {
        "Region": "G1",
        "RegionLabel": "سمنان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-10",
    },

    # G2 — اصفهان
    4: {
        "Region": "G2",
        "RegionLabel": "اصفهان",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-5",
    },
    5: {
        "Region": "G2",
        "RegionLabel": "اصفهان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-27",
    },
    6: {
        "Region": "G2",
        "RegionLabel": "اصفهان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-19",
    },
    7: {
        "Region": "G2",
        "RegionLabel": "اصفهان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-20",
    },
    8: {
        "Region": "G2",
        "RegionLabel": "اصفهان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-21",
    },

    # G3 — زنجان
    9: {
        "Region": "G3",
        "RegionLabel": "زنجان",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-7",
    },
    10: {
        "Region": "G3",
        "RegionLabel": "زنجان",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-8",
    },
    11: {
        "Region": "G3",
        "RegionLabel": "زنجان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-13",
    },
    12: {
        "Region": "G3",
        "RegionLabel": "زنجان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-22",
    },

    # G4 — اراک
    13: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-9",
    },
    14: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-4",
    },
    15: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-5",
    },
    16: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-6",
    },
    17: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-7",
    },
    18: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-14",
    },
    19: {
        "Region": "G4",
        "RegionLabel": "اراک",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-15",
    },

    # G5 — خراسان شمالی
    20: {
        "Region": "G5",
        "RegionLabel": "خراسان شمالی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-13",
    },
    21: {
        "Region": "G5",
        "RegionLabel": "خراسان شمالی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-3",
    },
    22: {
        "Region": "G5",
        "RegionLabel": "خراسان شمالی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-12",
    },
    23: {
        "Region": "G5",
        "RegionLabel": "خراسان شمالی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-23",
    },

    # G6 — خراسان رضوی
    24: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-21",
    },
    25: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-22",
    },
    26: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-25",
    },
    27: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-26",
    },
    28: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-27",
    },
    29: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-28",
    },
    30: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-1",
    },
    31: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-11",
    },
    32: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-16",
    },
    33: {
        "Region": "G6",
        "RegionLabel": "خراسان رضوی",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-18",
    },

    # G7 — فارس
    34: {
        "Region": "G7",
        "RegionLabel": "فارس",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-30",
    },
    35: {
        "Region": "G7",
        "RegionLabel": "فارس",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-30",
    },
    36: {
        "Region": "G7",
        "RegionLabel": "فارس",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-32",
    },

    # G8 — قم
    37: {
        "Region": "G8",
        "RegionLabel": "قم",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-33",
    },
    38: {
        "Region": "G8",
        "RegionLabel": "قم",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-2",
    },

    # G9 — یزد
    39: {
        "Region": "G9",
        "RegionLabel": "یزد",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-35",
    },
    40: {
        "Region": "G9",
        "RegionLabel": "یزد",
        "HarvestYear": 1394,
        "OriginalSampleName": "94-36",
    },

    # G10 — قزوین
    41: {
        "Region": "G10",
        "RegionLabel": "قزوین",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-8",
    },
    42: {
        "Region": "G10",
        "RegionLabel": "قزوین",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-9",
    },

    # G11 — لرستان
    43: {
        "Region": "G11",
        "RegionLabel": "لرستان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-24",
    },
    44: {
        "Region": "G11",
        "RegionLabel": "لرستان",
        "HarvestYear": 1404,
        "OriginalSampleName": "04-25",
    },
}


# ============================================================
# DATASET
# ============================================================

def prepare_dataset():

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("saffron.csv is empty.")

    metadata_columns = {
        "name",
        "group",
        "region_code",
        "harvest_year",
        "HarvestYear",
        "sample_id",
        "SampleId",
        "id",
        "ID",
    }

    spectral_columns = [
        c
        for c in df.columns
        if c not in metadata_columns
    ]

    if not spectral_columns:
        raise ValueError("No spectral columns found.")

    df["SampleId"] = pd.to_numeric(
        df["name"],
        errors="coerce",
    )

    if df["SampleId"].isna().any():

        bad_ids = (
            df.loc[
                df["SampleId"].isna(),
                "name",
            ]
            .tolist()
        )

        raise ValueError(
            "Could not parse SampleId from 'name'. "
            f"Invalid values: {bad_ids}"
        )

    df["SampleId"] = df["SampleId"].astype(int)

    dataset_ids = set(df["SampleId"].tolist())
    mapping_ids = set(SAMPLE_METADATA.keys())

    missing_mapping = dataset_ids - mapping_ids

    if missing_mapping:

        raise ValueError(
            "Missing metadata mapping for SampleId: "
            f"{sorted(missing_mapping)}"
        )

    # --------------------------------------------------------
    # Remove duplicate spectral measurements
    # --------------------------------------------------------

    duplicate_mask = (
        df[spectral_columns]
        .duplicated(keep="first")
    )

    removed_ids = (
        df.loc[
            duplicate_mask,
            "SampleId",
        ]
        .astype(int)
        .tolist()
    )

    removed_count = int(duplicate_mask.sum())

    df = (
        df[
            ~duplicate_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Official metadata mapping
    # --------------------------------------------------------

    df["group"] = (
        df["SampleId"]
        .map(
            {
                sample_id: info["Region"]
                for sample_id, info
                in SAMPLE_METADATA.items()
            }
        )
    )

    df["RegionLabel"] = (
        df["SampleId"]
        .map(
            {
                sample_id: info["RegionLabel"]
                for sample_id, info
                in SAMPLE_METADATA.items()
            }
        )
    )

    df["HarvestYear"] = (
        df["SampleId"]
        .map(
            {
                sample_id: info["HarvestYear"]
                for sample_id, info
                in SAMPLE_METADATA.items()
            }
        )
    )

    df["OriginalSampleName"] = (
        df["SampleId"]
        .map(
            {
                sample_id: info["OriginalSampleName"]
                for sample_id, info
                in SAMPLE_METADATA.items()
            }
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if df["group"].isna().any():

        bad_ids = (
            df.loc[
                df["group"].isna(),
                "SampleId",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing region mapping for SampleId: "
            f"{bad_ids}"
        )

    if df["HarvestYear"].isna().any():

        bad_ids = (
            df.loc[
                df["HarvestYear"].isna(),
                "SampleId",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing harvest-year mapping for SampleId: "
            f"{bad_ids}"
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(
        f"Duplicate spectral rows removed: "
        f"{removed_count}"
    )

    if removed_ids:
        print(
            "Removed SampleId:",
            removed_ids,
        )

    print()
    print("Harvest-year distribution:")

    print(
        df["HarvestYear"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Region × Year:")

    print(
        pd.crosstab(
            df["group"],
            df["HarvestYear"],
        )
        .sort_index()
    )

    return df, spectral_columns


# ============================================================
# SELECT TOP REGIONS
# ============================================================

def select_top_regions(df):

    counts = (
        df["group"]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )

    selected = (
        counts
        .head(TOP_REGIONS)
        .index
        .tolist()
    )

    result = pd.DataFrame(
        {
            "Group": counts.index,

            "RegionLabel": [
                REGION_LABELS.get(
                    group,
                    group,
                )
                for group in counts.index
            ],

            "Samples": counts.values,

            "SelectedForRegionalModel": [
                group in selected
                for group in counts.index
            ],
        }
    )

    return selected, result


# ============================================================
# FIND ENGINEERED FEATURES
# ============================================================

def find_engineered_features_path():

    candidates = [
        BASE_DIR / "engineered_features.csv",

        BASE_DIR
        / "outputs"
        / "feature_engineering"
        / "engineered_features.csv",

        BASE_DIR
        / "outputs"
        / "features"
        / "engineered_features.csv",

        BASE_DIR
        / "outputs"
        / "engineered_features.csv",

        BASE_DIR
        / "data"
        / "engineered_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Last resort: search project recursively.
    matches = list(
        BASE_DIR.rglob("engineered_features.csv")
    )

    # Exclude anything under virtual environments/cache folders.
    matches = [
        path
        for path in matches
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
    ]

    if matches:
        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return matches[0]

    return None


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(
    df,
    spectral_columns,
):

    engineered_path = (
        find_engineered_features_path()
    )

    if engineered_path is not None:

        print()
        print(
            "Engineered features found:"
        )
        print(
            f"  {engineered_path}"
        )

        features = pd.read_csv(
            engineered_path
        )

        feature_id_column = None

        for candidate in [
            "SampleId",
            "sample_id",
            "id",
            "ID",
        ]:

            if candidate in features.columns:

                feature_id_column = candidate
                break

        if feature_id_column is not None:

            if feature_id_column != "SampleId":

                features = features.rename(
                    columns={
                        feature_id_column:
                            "SampleId"
                    }
                )

            features["SampleId"] = pd.to_numeric(
                features["SampleId"],
                errors="coerce",
            )

            feature_columns = [
                c
                for c in features.columns
                if c != "SampleId"
                and pd.api.types.is_numeric_dtype(
                    features[c]
                )
            ]

            merged = (
                df[
                    [
                        "SampleId",
                        "group",
                        "RegionLabel",
                        "HarvestYear",
                    ]
                ]
                .merge(
                    features[
                        ["SampleId"]
                        + feature_columns
                    ],
                    on="SampleId",
                    how="inner",
                )
            )

            if (
                len(merged) == len(df)
                and len(feature_columns) > 0
            ):

                print(
                    "Engineered feature join:"
                    " SUCCESS"
                )

                return (
                    merged,
                    feature_columns,
                )

            print(
                "Engineered feature join:"
                " FAILED"
            )

            print(
                f"Dataset samples: {len(df)}"
            )

            print(
                f"Matched samples: {len(merged)}"
            )

            print(
                "Falling back to raw spectra."
            )

        else:

            print(
                "Engineered feature file found "
                "but no SampleId column exists."
            )

            print(
                "Falling back to raw spectra."
            )

    else:

        print()
        print(
            "Engineered features file not found."
        )

        print(
            "Falling back to raw spectra."
        )

    # --------------------------------------------------------
    # Fallback to raw spectrum
    # --------------------------------------------------------

    output = df[
        [
            "SampleId",
            "group",
            "RegionLabel",
            "HarvestYear",
        ]
    ].copy()

    numeric_spectral = (
        df[spectral_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    for column in numeric_spectral.columns:

        output[column] = (
            numeric_spectral[column]
        )

    return output, spectral_columns


# ============================================================
# REGIONAL BINARY MODEL
# ============================================================

def run_region_model(
    feature_df,
    feature_columns,
    region,
):

    model_data = feature_df.copy()

    model_data["Target"] = (
        model_data["group"]
        .astype(str)
        .eq(str(region))
        .astype(int)
    )

    y = model_data["Target"].to_numpy()

    X = (
        model_data[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    positive = int((y == 1).sum())
    negative = int((y == 0).sum())

    if positive < 2 or negative < 2:
        return pd.DataFrame()

    n_splits = min(
        N_SPLITS,
        positive,
        negative,
    )

    if n_splits < 2:
        return pd.DataFrame()

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []

    # ========================================================
    # SVM
    # ========================================================

    for top_k in TOP_K_FEATURES:

        k = min(
            top_k,
            X.shape[1],
        )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "select",
                    SelectKBest(
                        score_func=f_classif,
                        k=k,
                    ),
                ),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        scores = []

        for train_idx, test_idx in cv.split(X, y):

            model.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            pred = model.predict(
                X.iloc[test_idx]
            )

            scores.append(
                {
                    "Accuracy":
                        accuracy_score(
                            y[test_idx],
                            pred,
                        ),

                    "BalancedAccuracy":
                        balanced_accuracy_score(
                            y[test_idx],
                            pred,
                        ),

                    "MacroF1":
                        f1_score(
                            y[test_idx],
                            pred,
                            average="macro",
                            zero_division=0,
                        ),
                }
            )

        score_df = pd.DataFrame(scores)

        rows.append(
            {
                "Region": region,

                "RegionLabel":
                    REGION_LABELS.get(
                        region,
                        region,
                    ),

                "Model": "SVM",

                "TopK": k,

                "Samples": len(y),

                "PositiveSamples":
                    positive,

                "NegativeSamples":
                    negative,

                "AccuracyMean":
                    score_df["Accuracy"].mean(),

                "AccuracyStd":
                    score_df["Accuracy"].std(),

                "BalancedAccuracyMean":
                    score_df[
                        "BalancedAccuracy"
                    ].mean(),

                "BalancedAccuracyStd":
                    score_df[
                        "BalancedAccuracy"
                    ].std(),

                "MacroF1Mean":
                    score_df[
                        "MacroF1"
                    ].mean(),

                "MacroF1Std":
                    score_df[
                        "MacroF1"
                    ].std(),
            }
        )

    # ========================================================
    # XGBOOST
    # ========================================================

    try:

        from xgboost import XGBClassifier

        for top_k in TOP_K_FEATURES:

            k = min(
                top_k,
                X.shape[1],
            )

            model = Pipeline(
                [
                    (
                        "select",
                        SelectKBest(
                            score_func=f_classif,
                            k=k,
                        ),
                    ),
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=150,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            )

            scores = []

            for train_idx, test_idx in cv.split(X, y):

                model.fit(
                    X.iloc[train_idx],
                    y[train_idx],
                )

                pred = model.predict(
                    X.iloc[test_idx]
                )

                scores.append(
                    {
                        "Accuracy":
                            accuracy_score(
                                y[test_idx],
                                pred,
                            ),

                        "BalancedAccuracy":
                            balanced_accuracy_score(
                                y[test_idx],
                                pred,
                            ),

                        "MacroF1":
                            f1_score(
                                y[test_idx],
                                pred,
                                average="macro",
                                zero_division=0,
                            ),
                    }
                )

            score_df = pd.DataFrame(scores)

            rows.append(
                {
                    "Region": region,

                    "RegionLabel":
                        REGION_LABELS.get(
                            region,
                            region,
                        ),

                    "Model": "XGBoost",

                    "TopK": k,

                    "Samples": len(y),

                    "PositiveSamples":
                        positive,

                    "NegativeSamples":
                        negative,

                    "AccuracyMean":
                        score_df["Accuracy"].mean(),

                    "AccuracyStd":
                        score_df["Accuracy"].std(),

                    "BalancedAccuracyMean":
                        score_df[
                            "BalancedAccuracy"
                        ].mean(),

                    "BalancedAccuracyStd":
                        score_df[
                            "BalancedAccuracy"
                        ].std(),

                    "MacroF1Mean":
                        score_df[
                            "MacroF1"
                        ].mean(),

                    "MacroF1Std":
                        score_df[
                            "MacroF1"
                        ].std(),
                }
            )

    except ImportError:

        print(
            "XGBoost is not installed. "
            "Only SVM results will be generated."
        )

    return pd.DataFrame(rows)


# ============================================================
# FRESH PCA
# ============================================================

def build_fresh_pca(
    df,
    spectral_columns,
):

    X = (
        df[spectral_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    n_components = min(
        8,
        X.shape[0] - 1,
        X.shape[1],
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    scores = pca.fit_transform(X)

    pc_columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    output = df[
        [
            "SampleId",
            "group",
            "RegionLabel",
            "HarvestYear",
            "OriginalSampleName",
        ]
    ].copy()

    for index, column in enumerate(pc_columns):

        output[column] = (
            scores[:, index]
        )

    return output, pc_columns, pca


# ============================================================
# REGION / YEAR SIMILARITY
# ============================================================

def calculate_region_year_similarity(
    pca_df,
    pc_columns,
    output_dir,
    n_permutations=2000,
    random_state=42,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    df_1394 = (
        pca_df[
            pca_df["HarvestYear"] == 1394
        ]
        .copy()
    )

    df_1404 = (
        pca_df[
            pca_df["HarvestYear"] == 1404
        ]
        .copy()
    )

    print()
    print(
        f"1394 samples for similarity: "
        f"{len(df_1394)}"
    )

    print(
        f"1404 samples for similarity: "
        f"{len(df_1404)}"
    )

    if len(df_1394) == 0 or len(df_1404) == 0:

        raise ValueError(
            "Both harvest years must contain samples."
        )

    X_1394 = (
        df_1394[
            pc_columns
        ]
        .to_numpy(dtype=float)
    )

    X_1404 = (
        df_1404[
            pc_columns
        ]
        .to_numpy(dtype=float)
    )

    regions_1394 = (
        df_1394["group"]
        .astype(str)
        .to_numpy()
    )

    regions_1404 = (
        df_1404["group"]
        .astype(str)
        .to_numpy()
    )

    # ========================================================
    # PAIRWISE DISTANCE MATRIX
    # Rows = 1404
    # Columns = 1394
    # ========================================================

    distance_matrix = pairwise_distances(
        X_1404,
        X_1394,
        metric="euclidean",
    )

    # ========================================================
    # PAIRWISE TABLE
    # ========================================================

    pairwise_rows = []

    for i in range(len(df_1404)):

        for j in range(len(df_1394)):

            same_region = (
                regions_1404[i]
                ==
                regions_1394[j]
            )

            pairwise_rows.append(
                {
                    "SampleId_1404":
                        int(
                            df_1404.iloc[i][
                                "SampleId"
                            ]
                        ),

                    "Region_1404":
                        df_1404.iloc[i][
                            "RegionLabel"
                        ],

                    "Group_1404":
                        regions_1404[i],

                    "OriginalSampleName_1404":
                        df_1404.iloc[i][
                            "OriginalSampleName"
                        ],

                    "SampleId_1394":
                        int(
                            df_1394.iloc[j][
                                "SampleId"
                            ]
                        ),

                    "Region_1394":
                        df_1394.iloc[j][
                            "RegionLabel"
                        ],

                    "Group_1394":
                        regions_1394[j],

                    "OriginalSampleName_1394":
                        df_1394.iloc[j][
                            "OriginalSampleName"
                        ],

                    "Distance":
                        float(
                            distance_matrix[i, j]
                        ),

                    "SameRegion":
                        bool(same_region),
                }
            )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    pairwise_df.to_csv(
        Path(output_dir)
        / "region_year_pairwise_distances.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # OVERALL DISTANCES
    # ========================================================

    same_distances = (
        pairwise_df.loc[
            pairwise_df["SameRegion"],
            "Distance",
        ]
        .to_numpy(dtype=float)
    )

    other_distances = (
        pairwise_df.loc[
            ~pairwise_df["SameRegion"],
            "Distance",
        ]
        .to_numpy(dtype=float)
    )

    overall_same_mean = (
        float(np.mean(same_distances))
        if len(same_distances)
        else np.nan
    )

    overall_other_mean = (
        float(np.mean(other_distances))
        if len(other_distances)
        else np.nan
    )

    if (
        np.isfinite(overall_same_mean)
        and np.isfinite(overall_other_mean)
        and overall_other_mean > 0
    ):

        overall_ratio = (
            overall_same_mean
            /
            overall_other_mean
        )

    else:

        overall_ratio = np.nan

    # ========================================================
    # NEAREST NEIGHBOR
    # ========================================================

    nearest_rows = []

    nearest_same_flags = []

    for i in range(len(df_1404)):

        distances = distance_matrix[i]

        nearest_idx = int(
            np.argmin(distances)
        )

        nearest_region = (
            regions_1394[nearest_idx]
        )

        target_region = (
            regions_1404[i]
        )

        same_region = (
            nearest_region
            ==
            target_region
        )

        nearest_same_flags.append(
            same_region
        )

        nearest_rows.append(
            {
                "SampleId_1404":
                    int(
                        df_1404.iloc[i][
                            "SampleId"
                        ]
                    ),

                "Region_1404":
                    target_region,

                "RegionLabel_1404":
                    df_1404.iloc[i][
                        "RegionLabel"
                    ],

                "OriginalSampleName_1404":
                    df_1404.iloc[i][
                        "OriginalSampleName"
                    ],

                "NearestSampleId_1394":
                    int(
                        df_1394.iloc[
                            nearest_idx
                        ][
                            "SampleId"
                        ]
                    ),

                "NearestRegion_1394":
                    nearest_region,

                "NearestRegionLabel_1394":
                    df_1394.iloc[
                        nearest_idx
                    ][
                        "RegionLabel"
                    ],

                "NearestOriginalSampleName_1394":
                    df_1394.iloc[
                        nearest_idx
                    ][
                        "OriginalSampleName"
                    ],

                "NearestDistance":
                    float(
                        distances[
                            nearest_idx
                        ]
                    ),

                "SameRegion":
                    bool(same_region),
            }
        )

    nearest_df = pd.DataFrame(
        nearest_rows
    )

    nearest_df.to_csv(
        Path(output_dir)
        / "region_year_nearest_neighbors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    nearest_same_fraction = (
        float(
            np.mean(nearest_same_flags)
        )
        if nearest_same_flags
        else np.nan
    )

    # ========================================================
    # REGION-WISE SIMILARITY
    # ========================================================

    common_regions = sorted(
        set(regions_1394)
        .intersection(
            set(regions_1404)
        )
    )

    similarity_rows = []

    for region in common_regions:

        same_mask = (
            (pairwise_df["Group_1404"] == region)
            &
            (pairwise_df["Group_1394"] == region)
        )

        other_mask = (
            (pairwise_df["Group_1404"] == region)
            &
            (pairwise_df["Group_1394"] != region)
        )

        same_values = (
            pairwise_df.loc[
                same_mask,
                "Distance",
            ]
            .to_numpy(dtype=float)
        )

        other_values = (
            pairwise_df.loc[
                other_mask,
                "Distance",
            ]
            .to_numpy(dtype=float)
        )

        same_mean = (
            float(np.mean(same_values))
            if len(same_values)
            else np.nan
        )

        other_mean = (
            float(np.mean(other_values))
            if len(other_values)
            else np.nan
        )

        ratio = (
            same_mean / other_mean
            if (
                np.isfinite(same_mean)
                and np.isfinite(other_mean)
                and other_mean > 0
            )
            else np.nan
        )

        region_nearest = nearest_df[
            nearest_df["Region_1404"] == region
        ]

        nearest_fraction = (
            float(
                region_nearest[
                    "SameRegion"
                ].mean()
            )
            if len(region_nearest)
            else np.nan
        )

        similarity_rows.append(
            {
                "group":
                    region,

                "RegionLabel":
                    REGION_LABELS.get(
                        region,
                        region,
                    ),

                "N_1394":
                    int(
                        np.sum(
                            regions_1394
                            ==
                            region
                        )
                    ),

                "N_1404":
                    int(
                        np.sum(
                            regions_1404
                            ==
                            region
                        )
                    ),

                "SameRegionDistanceMean":
                    same_mean,

                "OtherRegionDistanceMean":
                    other_mean,

                "SameToOtherDistanceRatio":
                    ratio,

                "NearestSameRegionFraction":
                    nearest_fraction,

                "EvidenceRatioBelow1":
                    bool(
                        np.isfinite(ratio)
                        and ratio < 1.0
                    ),

                "EvidenceNearestAbove05":
                    bool(
                        np.isfinite(
                            nearest_fraction
                        )
                        and
                        nearest_fraction > 0.5
                    ),
            }
        )

    similarity_df = pd.DataFrame(
        similarity_rows
    )

    # ========================================================
    # PERMUTATION TEST
    # ========================================================

    actual_statistic = np.nan

    if (
        len(same_distances) > 0
        and len(other_distances) > 0
    ):

        actual_statistic = (
            float(
                np.mean(same_distances)
            )
            -
            float(
                np.mean(other_distances)
            )
        )

    rng = np.random.default_rng(
        random_state
    )

    permuted_statistics = []

    for _ in range(n_permutations):

        shuffled_regions_1394 = (
            rng.permutation(
                regions_1394
            )
        )

        same_mask_perm = (
            regions_1404[:, None]
            ==
            shuffled_regions_1394[None, :]
        )

        same_values_perm = (
            distance_matrix[
                same_mask_perm
            ]
        )

        other_values_perm = (
            distance_matrix[
                ~same_mask_perm
            ]
        )

        if (
            len(same_values_perm) == 0
            or len(other_values_perm) == 0
        ):
            continue

        stat_perm = (
            float(
                np.mean(
                    same_values_perm
                )
            )
            -
            float(
                np.mean(
                    other_values_perm
                )
            )
        )

        permuted_statistics.append(
            stat_perm
        )

    if (
        np.isfinite(actual_statistic)
        and len(permuted_statistics) > 0
    ):

        permuted_statistics = (
            np.asarray(
                permuted_statistics,
                dtype=float,
            )
        )

        p_value = (
            (
                np.sum(
                    permuted_statistics
                    <=
                    actual_statistic
                )
                + 1
            )
            /
            (
                len(
                    permuted_statistics
                )
                + 1
            )
        )

    else:

        p_value = np.nan

    # ========================================================
    # ADD GLOBAL P-VALUE
    # ========================================================

    similarity_df[
        "PermutationPValue"
    ] = p_value

    # ========================================================
    # OVERALL ROW
    # ========================================================

    overall_row = pd.DataFrame(
        [
            {
                "group": "ALL",

                "RegionLabel":
                    "همه مناطق",

                "N_1394":
                    int(len(df_1394)),

                "N_1404":
                    int(len(df_1404)),

                "SameRegionDistanceMean":
                    overall_same_mean,

                "OtherRegionDistanceMean":
                    overall_other_mean,

                "SameToOtherDistanceRatio":
                    overall_ratio,

                "NearestSameRegionFraction":
                    nearest_same_fraction,

                "PermutationPValue":
                    p_value,

                "EvidenceRatioBelow1":
                    bool(
                        np.isfinite(
                            overall_ratio
                        )
                        and
                        overall_ratio < 1.0
                    ),

                "EvidenceNearestAbove05":
                    bool(
                        np.isfinite(
                            nearest_same_fraction
                        )
                        and
                        nearest_same_fraction > 0.5
                    ),
            }
        ]
    )

    similarity_output_df = pd.concat(
        [
            overall_row,
            similarity_df,
        ],
        ignore_index=True,
    )

    similarity_output_df.to_csv(
        Path(output_dir)
        / "region_year_similarity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SCIENTIFIC INTERPRETATION
    # ========================================================

    if (
        np.isfinite(overall_ratio)
        and overall_ratio < 1.0
        and np.isfinite(
            nearest_same_fraction
        )
        and nearest_same_fraction > 0.5
        and np.isfinite(p_value)
        and p_value < 0.05
    ):

        interpretation = (
            "شواهد اولیه به نفع پایداری "
            "الگوی منطقه‌ای بین دو سال "
            "برداشت مشاهده شد."
        )

    elif (
        np.isfinite(overall_ratio)
        and overall_ratio < 1.0
    ):

        interpretation = (
            "شباهت اولیه بین نمونه‌های یک منطقه "
            "در دو سال مشاهده شد؛ اما شواهد آماری "
            "کافی برای نتیجه‌گیری قطعی وجود ندارد."
        )

    else:

        interpretation = (
            "در این داده‌ها شواهد کافی برای پایداری "
            "الگوی منطقه‌ای بین دو سال مشاهده نشد."
        )

    # ========================================================
    # CONSOLE
    # ========================================================

    print()
    print("-" * 72)
    print("REGION × YEAR SIMILARITY")
    print("-" * 72)

    print(
        f"Same-region mean distance : "
        f"{overall_same_mean:.4f}"
    )

    print(
        f"Other-region mean distance: "
        f"{overall_other_mean:.4f}"
    )

    print(
        f"Same / Other ratio        : "
        f"{overall_ratio:.4f}"
    )

    print(
        f"Nearest same-region frac. : "
        f"{nearest_same_fraction:.4f}"
    )

    print(
        f"Permutation p-value       : "
        f"{p_value:.4f}"
    )

    print()
    print(
        f"Interpretation: "
        f"{interpretation}"
    )

    return {
        "similarity_df":
            similarity_output_df,

        "pairwise_df":
            pairwise_df,

        "nearest_df":
            nearest_df,

        "same_region_mean_distance":
            overall_same_mean,

        "other_region_mean_distance":
            overall_other_mean,

        "same_other_ratio":
            overall_ratio,

        "nearest_same_region_fraction":
            nearest_same_fraction,

        "permutation_p_value":
            p_value,

        "interpretation":
            interpretation,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "REGION-WISE / YEAR-WISE ANALYSIS"
    )
    print("=" * 72)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df, spectral_columns = (
        prepare_dataset()
    )

    print()
    print(
        f"Independent samples: "
        f"{len(df)}"
    )

    print(
        f"Spectral points: "
        f"{len(spectral_columns)}"
    )

    # ========================================================
    # REGION SELECTION
    # ========================================================

    selected_regions, region_table = (
        select_top_regions(df)
    )

    print()
    print("Top regions:")

    for region in selected_regions:

        count = int(
            (
                df["group"] == region
            ).sum()
        )

        print(
            f"  {region} - "
            f"{REGION_LABELS.get(region, region)} "
            f"({count} samples)"
        )

    region_table.to_csv(
        OUTPUT_DIR
        / "region_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FEATURES
    # ========================================================

    feature_df, feature_columns = (
        load_features(
            df,
            spectral_columns,
        )
    )

    print()
    print(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # REGIONAL MODELS
    # ========================================================

    all_results = []

    for region in selected_regions:

        print()
        print("-" * 72)

        print(
            f"Regional model: "
            f"{region} - "
            f"{REGION_LABELS.get(region, region)}"
        )

        result = run_region_model(
            feature_df,
            feature_columns,
            region,
        )

        if result.empty:

            print(
                "No valid regional model result."
            )

            continue

        all_results.append(result)

        best_index = (
            result[
                "BalancedAccuracyMean"
            ]
            .idxmax()
        )

        best = result.loc[
            best_index
        ]

        print(
            f"Best model: "
            f"{best['Model']} / "
            f"TopK={int(best['TopK'])}"
        )

        print(
            f"Accuracy: "
            f"{best['AccuracyMean']:.4f}"
            f" ± "
            f"{best['AccuracyStd']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{best['BalancedAccuracyMean']:.4f}"
            f" ± "
            f"{best['BalancedAccuracyStd']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{best['MacroF1Mean']:.4f}"
            f" ± "
            f"{best['MacroF1Std']:.4f}"
        )

    if all_results:

        regional_results = (
            pd.concat(
                all_results,
                ignore_index=True,
            )
            .sort_values(
                "BalancedAccuracyMean",
                ascending=False,
            )
        )

        regional_results.to_csv(
            OUTPUT_DIR
            / "regional_model_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    else:

        regional_results = (
            pd.DataFrame()
        )

    # ========================================================
    # FRESH PCA
    # ========================================================

    print()
    print("-" * 72)

    print(
        "Building fresh PCA for "
        "Region × Year..."
    )

    pca_df, pc_columns, pca = (
        build_fresh_pca(
            df,
            spectral_columns,
        )
    )

    explained = float(
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"PCA components: "
        f"{len(pc_columns)}"
    )

    print(
        f"Cumulative explained variance: "
        f"{explained:.4f}"
    )

    pca_df.to_csv(
        OUTPUT_DIR
        / "region_year_pca_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # REGION × YEAR SIMILARITY
    # ========================================================

    similarity_result = (
        calculate_region_year_similarity(
            pca_df,
            pc_columns,
            OUTPUT_DIR,
            n_permutations=PERMUTATIONS,
            random_state=RANDOM_STATE,
        )
    )

    similarity_df = (
        similarity_result[
            "similarity_df"
        ]
    )

    pairwise_df = (
        similarity_result[
            "pairwise_df"
        ]
    )

    nearest_df = (
        similarity_result[
            "nearest_df"
        ]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    common_regions = int(
        (
            df.groupby("group")[
                "HarvestYear"
            ]
            .nunique()
            >= 2
        )
        .sum()
    )

    summary = pd.DataFrame(
        [
            {
                "IndependentSamples":
                    len(df),

                "SelectedRegions":
                    len(selected_regions),

                "SelectedRegionList":
                    ",".join(
                        selected_regions
                    ),

                "CommonRegionsBothYears":
                    common_regions,

                "FeatureCount":
                    len(feature_columns),

                "PCCount":
                    len(pc_columns),

                "PCAExplainedVariance":
                    explained,

                "SameRegionMeanDistance":
                    similarity_result[
                        "same_region_mean_distance"
                    ],

                "OtherRegionMeanDistance":
                    similarity_result[
                        "other_region_mean_distance"
                    ],

                "SameOtherDistanceRatio":
                    similarity_result[
                        "same_other_ratio"
                    ],

                "NearestSameRegionFraction":
                    similarity_result[
                        "nearest_same_region_fraction"
                    ],

                "PermutationPValue":
                    similarity_result[
                        "permutation_p_value"
                    ],

                "SimilarityRows":
                    len(similarity_df),

                "PairwiseRows":
                    len(pairwise_df),

                "NearestNeighborRows":
                    len(nearest_df),

                "Interpretation":
                    similarity_result[
                        "interpretation"
                    ],
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "analysis_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print("ANALYSIS FINISHED")
    print("=" * 72)

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    for file in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if file.is_file():

            print(
                f"  - {file.name}"
            )


if __name__ == "__main__":
    main()