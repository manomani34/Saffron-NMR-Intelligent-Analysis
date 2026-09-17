from pathlib import Path
from datetime import datetime
import json
import contextlib
import io

import pandas as pd
import streamlit as st

from src.load_data import load_dataset
from src.prepare_data import prepare_dataset


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"

DATASET_PATH = DATA_DIR / "raw" / "saffron.csv"

ROBUST_SUMMARY_PATH = (
    BASE_DIR / "robust_evaluation_summary.csv"
)

ROBUST_RESULTS_PATH = (
    BASE_DIR / "robust_evaluation_results.csv"
)

NOVELTY_PATH = (
    BASE_DIR / "novelty_detection_results.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR / "sample_predictions.csv"
)

DECISION_PATH = (
    BASE_DIR / "decision_engine_results.csv"
)

SHAP_PATH = (
    BASE_DIR / "shap_feature_importance.csv"
)

FINAL_REPORT_PATH = (
    BASE_DIR / "final_report.csv"
)

RUN_OUTPUT_PATH = (
    BASE_DIR / "run_output.txt"
)

README_PATH = (
    BASE_DIR / "README.md"
)

CONCLUSION_PATH = (
    DOCS_DIR / "conclusion_fa.md"
)

CROSS_YEAR_PATH = (
    OUTPUTS_DIR / "cross_year_origin_results.csv"
)

CROSS_YEAR_SUMMARY_PATH = (
    OUTPUTS_DIR / "cross_year_origin_summary.csv"
)

YEAR_COUNTS_PATH = (
    OUTPUTS_DIR / "harvest_year_counts.csv"
)

REGION_YEAR_PATH = (
    OUTPUTS_DIR / "region_by_harvest_year.csv"
)

REGION_YEAR_COVERAGE_PATH = (
    OUTPUTS_DIR / "region_year_coverage.csv"
)

COLOR_DASHBOARD_JSON = (
    BASE_DIR
    / "reports"
    / "color_dashboard"
    / "color_dashboard_result.json"
)

COLOR_DASHBOARD_CSV = (
    BASE_DIR
    / "reports"
    / "color_dashboard"
    / "color_dashboard_samples.csv"
)

PCA_SCORES_PLOT = (
    BASE_DIR / "pca_scores.png"
)

PCA_SCREE_PLOT = (
    BASE_DIR / "pca_scree_plot.png"
)

NOVELTY_PLOT = (
    BASE_DIR / "novelty_detection.png"
)

SHAP_PLOT = (
    BASE_DIR / "shap_summary.png"
)

YEAR_PCA_PLOT = (
    OUTPUTS_DIR / "year_pca_plot.png"
)

YEAR_REGION_PCA_PLOT = (
    OUTPUTS_DIR / "year_region_pca_plot.png"
)


# ============================================================
# FILE IO
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path):

    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(path):

    path = Path(path)

    if not path.exists():
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


@st.cache_data(show_spinner=False)
def read_text(path):

    path = Path(path)

    if not path.exists():
        return ""

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "cp1256",
        "cp1252",
    ):

        try:

            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError:

            continue

        except Exception:

            return ""

    return ""


# ============================================================
# HELPERS
# ============================================================

def file_time(path):

    path = Path(path)

    if not path.exists():
        return "وجود ندارد"

    try:

        return datetime.fromtimestamp(
            path.stat().st_mtime
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    except Exception:

        return "نامشخص"


def fmt(
    value,
    decimals=4,
):

    try:

        return (
            f"{float(value):.{decimals}f}"
        )

    except Exception:

        return "—"


def fmt_pm(
    mean,
    std,
    decimals=4,
):

    if pd.isna(mean):
        return "—"

    if pd.isna(std):
        return fmt(
            mean,
            decimals,
        )

    return (
        f"{fmt(mean, decimals)} ± "
        f"{fmt(std, decimals)}"
    )


def normalize_bool(value):

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "بله",
    }


# ============================================================
# OFFICIAL PROJECT METADATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_prepared_metadata():
    """
    Load the dataset through the project's official loader so that
    derived metadata such as region_code is created exactly like
    in the main analysis pipeline.

    Then run the official prepare_dataset() function so that:
        - duplicate spectral measurements are removed
        - official region mapping is applied
        - official harvest-year mapping is applied
    """

    try:
        # IMPORTANT:
        # Do not use pd.read_csv() directly here.
        # load_dataset() creates derived columns such as region_code.
        df = load_dataset(
            str(DATASET_PATH)
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Use the official project preparation pipeline.
        _, _, metadata = prepare_dataset(df)

        if metadata is None:
            return pd.DataFrame()

        return metadata.copy()

    except Exception as exc:
        # Keep the dashboard alive, but expose the problem clearly.
        st.warning(
            f"خطا در آماده‌سازی متادیتای دیتاست: {exc}"
        )
        return pd.DataFrame()


# ============================================================
# DATASET INFORMATION
# ============================================================

@st.cache_data(
    show_spinner=False
)
def get_dataset_info():

    result = {
        "records": 0,
        "independent": 0,
        "spectral_points": 0,
        "groups": 0,
        "missing": 0,
        "invalid": 0,
        "group_counts": pd.DataFrame(),
        "year_counts": pd.DataFrame(),
        "years": [],
        "region_year": pd.DataFrame(),
        "sample_metadata": pd.DataFrame(),
        "min_ppm": None,
        "max_ppm": None,
    }

    # --------------------------------------------------------
    # RAW DATASET
    # --------------------------------------------------------

    df = read_csv(
        DATASET_PATH
    )

    if df.empty:
        return result

    result[
        "records"
    ] = len(df)

    # --------------------------------------------------------
    # METADATA FROM THE OFFICIAL PIPELINE
    # --------------------------------------------------------

    metadata = load_prepared_metadata()

    if metadata.empty:
        return result

    result[
        "sample_metadata"
    ] = metadata

    # --------------------------------------------------------
    # SPECTRAL COLUMNS
    # --------------------------------------------------------

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
        "OriginalSampleName",
        "Region",
        "RegionLabel",
        "RegionCode",
    }

    spectral_columns = [
        column
        for column in df.columns
        if str(column)
        not in metadata_columns
    ]

    result[
        "spectral_points"
    ] = len(
        spectral_columns
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    result[
        "missing"
    ] = int(
        df.isna()
        .sum()
        .sum()
    )

    # --------------------------------------------------------
    # SPECTRAL VALIDATION
    # --------------------------------------------------------

    if spectral_columns:

        numeric = df[
            spectral_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        result[
            "invalid"
        ] = int(
            numeric.isna()
            .sum()
            .sum()
        )

        ppm = pd.to_numeric(
            pd.Index(
                spectral_columns
            ),
            errors="coerce",
        )

        ppm = ppm[
            ~pd.isna(ppm)
        ]

        if len(ppm) > 0:

            result[
                "min_ppm"
            ] = float(
                ppm.min()
            )

            result[
                "max_ppm"
            ] = float(
                ppm.max()
            )

    # ========================================================
    # GROUP DISTRIBUTION
    # ========================================================

    if "Group" in metadata.columns:

        group_counts = (
            metadata[
                "Group"
            ]
            .astype(str)
            .value_counts()
            .sort_index()
            .rename_axis("Group")
            .reset_index(
                name="Samples"
            )
        )

    elif "group" in metadata.columns:

        group_counts = (
            metadata[
                "group"
            ]
            .astype(str)
            .value_counts()
            .sort_index()
            .rename_axis("Group")
            .reset_index(
                name="Samples"
            )
        )

    else:

        group_counts = pd.DataFrame()

    result[
        "group_counts"
    ] = group_counts

    result[
        "groups"
    ] = len(
        group_counts
    )

    # ========================================================
    # HARVEST YEAR DISTRIBUTION
    # ========================================================

    if "HarvestYear" in metadata.columns:

        year_values = pd.to_numeric(
            metadata[
                "HarvestYear"
            ],
            errors="coerce",
        )

    elif "harvest_year" in metadata.columns:

        year_values = pd.to_numeric(
            metadata[
                "harvest_year"
            ],
            errors="coerce",
        )

    else:

        year_values = pd.Series(
            dtype="float64"
        )

    valid_years = (
        year_values
        .dropna()
        .astype(int)
    )

    if not valid_years.empty:

        year_counts = (
            valid_years
            .value_counts()
            .sort_index()
            .rename_axis(
                "HarvestYear"
            )
            .reset_index(
                name="Samples"
            )
        )

        result[
            "year_counts"
        ] = year_counts

        result[
            "years"
        ] = (
            year_counts[
                "HarvestYear"
            ]
            .astype(int)
            .tolist()
        )

    # ========================================================
    # REGION × HARVEST YEAR
    # ========================================================

    if (
        "Group" in metadata.columns
        and "HarvestYear" in metadata.columns
    ):

        region_year = pd.crosstab(
            metadata[
                "Group"
            ].astype(str),
            pd.to_numeric(
                metadata[
                    "HarvestYear"
                ],
                errors="coerce",
            ),
        )

        region_year = (
            region_year
            .sort_index()
            .sort_index(
                axis=1
            )
        )

        result[
            "region_year"
        ] = region_year

    # ========================================================
    # INDEPENDENT SAMPLE COUNT
    # ========================================================

    # prepare_dataset() has already removed the confirmed
    # duplicate measurement.
    result[
        "independent"
    ] = len(metadata)

    return result


# ============================================================
# ORIGIN MODEL METRICS
# ============================================================

def get_origin_metrics():

    result = {
        "top_k": "—",
        "accuracy": "—",
        "accuracy_std": "—",
        "ba": "—",
        "ba_std": "—",
        "f1": "—",
        "f1_std": "—",
        "status": "NOT RELIABLE",
    }

    robust = read_csv(
        ROBUST_SUMMARY_PATH
    )

    required = {
        "TopK",
        "AccuracyMean",
        "AccuracyStd",
        "BalancedAccuracyMean",
        "BalancedAccuracyStd",
        "F1MacroMean",
        "F1MacroStd",
    }

    if robust.empty:
        return result

    if not required.issubset(
        robust.columns
    ):
        return result

    robust = robust.copy()

    for column in required:

        robust[column] = pd.to_numeric(
            robust[column],
            errors="coerce",
        )

    robust = robust.dropna(
        subset=[
            "BalancedAccuracyMean"
        ]
    )

    if robust.empty:
        return result

    row = robust.loc[
        robust[
            "BalancedAccuracyMean"
        ].idxmax()
    ]

    result[
        "top_k"
    ] = str(
        int(
            row["TopK"]
        )
    )

    result[
        "accuracy"
    ] = fmt(
        row["AccuracyMean"]
    )

    result[
        "accuracy_std"
    ] = fmt(
        row["AccuracyStd"]
    )

    result[
        "ba"
    ] = fmt(
        row[
            "BalancedAccuracyMean"
        ]
    )

    result[
        "ba_std"
    ] = fmt(
        row[
            "BalancedAccuracyStd"
        ]
    )

    result[
        "f1"
    ] = fmt(
        row["F1MacroMean"]
    )

    result[
        "f1_std"
    ] = fmt(
        row["F1MacroStd"]
    )

    result[
        "status"
    ] = (
        "ABOVE CHANCE / NOT RELIABLE"
    )

    return result


# ============================================================
# OOD
# ============================================================

def get_ood_count():

    novelty = read_csv(
        NOVELTY_PATH
    )

    if novelty.empty:
        return 0

    for column in (
        "NoveltyStatus",
        "Status",
    ):

        if column in novelty.columns:

            values = (
                novelty[column]
                .astype(str)
            )

            return int(
                values
                .str.contains(
                    "NOVEL|OOD",
                    case=False,
                    regex=True,
                )
                .sum()
            )

    return 0


# ============================================================
# PAGE STYLE
# ============================================================

def apply_page_style():

    st.markdown(
        """
        <style>

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            direction: rtl !important;
        }

        .main,
        .block-container {
            direction: rtl !important;
            text-align: right !important;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        body,
        p,
        span,
        div,
        label,
        button,
        input,
        textarea {
            font-family:
                Tahoma,
                Arial,
                sans-serif !important;
        }

        [data-testid="stMarkdownContainer"] {
            direction: rtl !important;
            text-align: right !important;
        }

        [data-testid="stMarkdownContainer"] p {
            line-height: 1.9;
        }

        [data-testid="stMetric"] {
            direction: rtl !important;
            text-align: center !important;
            background: white;
            border: 1px solid #e4eae6;
            border-radius: 16px;
            padding: 15px 10px;
            min-height: 110px;
        }

        [data-testid="stMetricValue"] {
            direction: ltr !important;
            text-align: center !important;
            font-weight: 800 !important;
        }

        [data-testid="stDataFrame"] {
            direction: ltr !important;
        }

        [data-testid="stImage"] {
            direction: ltr !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# REFRESH
# ============================================================

def show_refresh_button():

    if st.sidebar.button(
        "🔄 بروزرسانی داشبورد",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()
