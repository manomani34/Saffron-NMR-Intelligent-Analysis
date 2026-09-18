from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ==============================================================================
# PATHS
# ==============================================================================

ROOT = Path(__file__).resolve().parents[1]

EXCEL_PATH = ROOT / "data" / "raw" / "hplc.xlsx"
MAPPING_PATH = ROOT / "data" / "mapping" / "sample_mapping.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SETTINGS
# ==============================================================================

WAVELENGTHS = [250, 308, 440]

DUPLICATE_PAIRS = [
    (1, 2),
    (24, 25),
    (28, 29),
    (34, 35),
]

MAX_TIME = 30.0

ALS_LAMBDA = 1e5
ALS_P = 0.01
ALS_ITERATIONS = 10


# ==============================================================================
# MAPPING
# ==============================================================================

def detect_column(
    columns,
    candidates: set[str],
):
    for col in columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if normalized in candidates:
            return col

    return None


def load_mapping() -> pd.DataFrame:

    mapping = pd.read_csv(
        MAPPING_PATH
    )

    sample_col = detect_column(
        mapping.columns,
        {
            "sampleid",
            "sample",
            "number",
        },
    )

    group_col = detect_column(
        mapping.columns,
        {
            "group",
            "class",
            "region",
        },
    )

    year_col = detect_column(
        mapping.columns,
        {
            "year",
            "harvestyear",
            "harvest_year",
        },
    )

    if sample_col is None:
        raise ValueError(
            "Sample ID column not found in mapping CSV."
        )

    if group_col is None:
        raise ValueError(
            "Group column not found in mapping CSV."
        )

    result = pd.DataFrame()

    result["SampleId"] = pd.to_numeric(
        mapping[sample_col],
        errors="coerce",
    )

    result["ActualClass"] = (
        mapping[group_col]
        .astype(str)
        .str.strip()
    )

    if year_col is not None:
        result["Year"] = mapping[year_col]
    else:
        result["Year"] = ""

    result = result.dropna(
        subset=["SampleId"]
    )

    result["SampleId"] = (
        result["SampleId"]
        .astype(int)
    )

    return result


# ==============================================================================
# INDEPENDENT SAMPLE IDS
# ==============================================================================

def get_independent_sample_ids(
    mapping: pd.DataFrame,
) -> list[int]:

    duplicate_second_ids = {
        second
        for _, second in DUPLICATE_PAIRS
    }

    sample_ids = sorted(
        mapping["SampleId"]
        .astype(int)
        .unique()
        .tolist()
    )

    return [
        sid
        for sid in sample_ids
        if sid not in duplicate_second_ids
    ]


# ==============================================================================
# EXCEL HEADER
# ==============================================================================

def find_header_row(
    raw: pd.DataFrame,
) -> int:

    for row_idx in range(
        min(len(raw), 30)
    ):
        values = [
            str(v).strip().lower()
            for v in raw.iloc[row_idx].tolist()
        ]

        if (
            "number" in values
            and "group" in values
        ):
            return row_idx

    raise ValueError(
        "Could not find HPLC header row."
    )


# ==============================================================================
# LOAD HPLC SHEET
# ==============================================================================

def get_sheet_name(
    wavelength: int,
) -> str:

    if wavelength == 250:
        return "picrocrocin 250"

    if wavelength == 308:
        return "safranal 308"

    if wavelength == 440:
        return "Crocin 440"

    raise ValueError(
        f"Unsupported wavelength: {wavelength}"
    )


def load_sheet(
    wavelength: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:

    sheet_name = get_sheet_name(
        wavelength
    )

    raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name=sheet_name,
        header=None,
    )

    header_row = find_header_row(
        raw
    )

    header = raw.iloc[
        header_row
    ].copy()

    data = raw.iloc[
        header_row + 1 :
    ].copy()

    data.columns = header

    number_col = None

    for col in data.columns:
        text = str(col).strip().lower()

        if text == "number":
            number_col = col
            break

    if number_col is None:
        raise ValueError(
            f"Number column not found in {sheet_name}"
        )

    # ------------------------------------------------------------------
    # Numeric time columns
    # ------------------------------------------------------------------

    time_columns = []

    for col in data.columns:
        try:
            value = float(col)

            if (
                value > 0
                and value <= MAX_TIME
            ):
                time_columns.append(col)

        except Exception:
            continue

    if not time_columns:
        raise ValueError(
            f"No numeric time columns found in {sheet_name}"
        )

    time_columns = sorted(
        time_columns,
        key=lambda c: float(c),
    )

    sample_ids = pd.to_numeric(
        data[number_col],
        errors="coerce",
    )

    valid = sample_ids.notna()

    data = data.loc[
        valid
    ].copy()

    sample_ids = (
        sample_ids.loc[valid]
        .astype(int)
        .to_numpy()
    )

    matrix = (
        data[time_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    # ------------------------------------------------------------------
    # Missing point interpolation
    # ------------------------------------------------------------------

    matrix_df = pd.DataFrame(
        matrix
    )

    matrix_df = (
        matrix_df
        .interpolate(
            axis=1,
            limit_direction="both",
        )
        .fillna(0.0)
    )

    matrix = matrix_df.to_numpy(
        dtype=float
    )

    time = np.asarray(
        [
            float(c)
            for c in time_columns
        ],
        dtype=float,
    )

    result = pd.DataFrame(
        matrix,
        index=sample_ids,
        columns=time,
    )

    return result, time


# ==============================================================================
# ALS BASELINE
# ==============================================================================

def baseline_asls(
    y: np.ndarray,
    lam: float = ALS_LAMBDA,
    p: float = ALS_P,
    n_iter: int = ALS_ITERATIONS,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(y)

    if n < 3:
        return np.zeros_like(y)

    # Second difference operator
    d = diags(
        [1.0, -2.0, 1.0],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )

    dtd = d.T @ d

    weights = np.ones(n)

    for _ in range(n_iter):

        w = diags(
            weights,
            offsets=0,
            shape=(n, n),
            format="csc",
        )

        z = spsolve(
            w + lam * dtd,
            weights * y,
        )

        weights = np.where(
            y > z,
            p,
            1.0 - p,
        )

    return np.asarray(z)


def asls_snv(
    y: np.ndarray,
) -> np.ndarray:

    baseline = baseline_asls(
        y
    )

    corrected = (
        np.asarray(y, dtype=float)
        - baseline
    )

    mean = np.mean(
        corrected
    )

    std = np.std(
        corrected
    )

    if std < 1e-12:
        return np.zeros_like(
            corrected
        )

    return (
        corrected - mean
    ) / std


# ==============================================================================
# SAMPLE METRICS
# ==============================================================================

def calculate_sample_metrics(
    matrix: pd.DataFrame,
    time: np.ndarray,
    wavelength: int,
) -> pd.DataFrame:

    rows = []

    for sample_id, row in matrix.iterrows():

        y = row.to_numpy(
            dtype=float
        )

        rows.append(
            {
                "SampleId": int(sample_id),
                "Wavelength": wavelength,

                "AUC": float(
                    np.trapezoid(
                        y,
                        time,
                    )
                ),

                "AbsAUC": float(
                    np.trapezoid(
                        np.abs(y),
                        time,
                    )
                ),

                "MaxIntensity": float(
                    np.max(y)
                ),

                "MinIntensity": float(
                    np.min(y)
                ),

                "MeanIntensity": float(
                    np.mean(y)
                ),

                "MedianIntensity": float(
                    np.median(y)
                ),

                "StdIntensity": float(
                    np.std(y)
                ),

                "RangeIntensity": float(
                    np.max(y)
                    - np.min(y)
                ),

                "P95Intensity": float(
                    np.percentile(y, 95)
                ),

                "P05Intensity": float(
                    np.percentile(y, 5)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# SUMMARY HELPERS
# ==============================================================================

def summarize_by_group(
    metrics: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    merged = metrics.merge(
        mapping[
            [
                "SampleId",
                "ActualClass",
                "Year",
            ]
        ],
        on="SampleId",
        how="left",
    )

    numeric_cols = [
        "AUC",
        "AbsAUC",
        "MaxIntensity",
        "MinIntensity",
        "MeanIntensity",
        "MedianIntensity",
        "StdIntensity",
        "RangeIntensity",
        "P95Intensity",
        "P05Intensity",
    ]

    summary = (
        merged
        .groupby(
            [
                "Wavelength",
                "ActualClass",
            ],
            dropna=False,
        )[numeric_cols]
        .agg(
            [
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    # Flatten multi-index column names
    new_columns = []

    for col in summary.columns:

        if isinstance(col, tuple):

            if col[1] == "":
                new_columns.append(
                    str(col[0])
                )
            else:
                new_columns.append(
                    f"{col[0]}_{col[1]}"
                )

        else:
            new_columns.append(
                str(col)
            )

    summary.columns = new_columns

    return summary


def summarize_by_year(
    metrics: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    merged = metrics.merge(
        mapping[
            [
                "SampleId",
                "ActualClass",
                "Year",
            ]
        ],
        on="SampleId",
        how="left",
    )

    numeric_cols = [
        "AUC",
        "AbsAUC",
        "MaxIntensity",
        "MinIntensity",
        "MeanIntensity",
        "MedianIntensity",
        "StdIntensity",
        "RangeIntensity",
        "P95Intensity",
        "P05Intensity",
    ]

    return (
        merged
        .groupby(
            [
                "Wavelength",
                "Year",
            ],
            dropna=False,
        )[numeric_cols]
        .agg(
            [
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )


# ==============================================================================
# PCA
# ==============================================================================

def prepare_processed_matrix(
    matrix: pd.DataFrame,
    sample_ids: list[int],
) -> np.ndarray:

    selected = (
        matrix
        .loc[sample_ids]
        .to_numpy(dtype=float)
    )

    processed = []

    for row in selected:
        processed.append(
            asls_snv(row)
        )

    return np.asarray(
        processed,
        dtype=float,
    )


def pca_analysis(
    X: np.ndarray,
    sample_ids: list[int],
    mapping: pd.DataFrame,
    modality: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    n_samples, n_features = X.shape

    n_components = min(
        10,
        n_samples - 1,
        n_features,
    )

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(
        X
    )

    explained = (
        pca.explained_variance_ratio_
    )

    # ------------------------------------------------------------------
    # Sample scores
    # ------------------------------------------------------------------

    score_df = pd.DataFrame(
        {
            "SampleId": sample_ids,
            "Modality": modality,
        }
    )

    merged_mapping = (
        mapping[
            [
                "SampleId",
                "ActualClass",
                "Year",
            ]
        ]
        .drop_duplicates(
            subset=["SampleId"]
        )
    )

    score_df = score_df.merge(
        merged_mapping,
        on="SampleId",
        how="left",
    )

    for i in range(
        n_components
    ):

        score_df[
            f"PC{i + 1}"
        ] = scores[:, i]

    # ------------------------------------------------------------------
    # PCA variance
    # ------------------------------------------------------------------

    variance_rows = []

    cumulative = 0.0

    for i, value in enumerate(
        explained,
        start=1,
    ):

        cumulative += float(
            value
        )

        variance_rows.append(
            {
                "Modality": modality,
                "PC": i,
                "ExplainedVariance": float(value),
                "CumulativeVariance": cumulative,
            }
        )

    variance_df = pd.DataFrame(
        variance_rows
    )

    return (
        score_df,
        variance_df,
    )


# ==============================================================================
# ETA SQUARED
# ==============================================================================

def eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    overall_mean = np.mean(
        values
    )

    total_ss = np.sum(
        (values - overall_mean) ** 2
    )

    if total_ss < 1e-12:
        return 0.0

    between_ss = 0.0

    for label in np.unique(
        labels
    ):

        group_values = values[
            labels == label
        ]

        if len(group_values) == 0:
            continue

        group_mean = np.mean(
            group_values
        )

        between_ss += (
            len(group_values)
            * (
                group_mean
                - overall_mean
            ) ** 2
        )

    return float(
        between_ss / total_ss
    )


# ==============================================================================
# PCA GROUP / YEAR SUMMARY
# ==============================================================================

def pca_structure_summary(
    score_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for modality in sorted(
        score_df["Modality"].unique()
    ):

        current = score_df[
            score_df["Modality"] == modality
        ].copy()

        pc_columns = [
            col
            for col in current.columns
            if str(col).startswith("PC")
        ]

        for pc in pc_columns:

            values = current[
                pc
            ].to_numpy(
                dtype=float
            )

            geography_eta = eta_squared(
                values,
                current[
                    "ActualClass"
                ].astype(str).to_numpy(),
            )

            year_eta = eta_squared(
                values,
                current[
                    "Year"
                ].astype(str).to_numpy(),
            )

            rows.append(
                {
                    "Modality": modality,
                    "PC": pc,
                    "GeographyEtaSquared": geography_eta,
                    "YearEtaSquared": year_eta,
                }
            )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# SILHOUETTE: GEOGRAPHY VS YEAR
# ==============================================================================

def calculate_label_silhouettes(
    score_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for modality in sorted(
        score_df["Modality"].unique()
    ):

        current = score_df[
            score_df["Modality"] == modality
        ].copy()

        pc_columns = [
            col
            for col in current.columns
            if str(col).startswith("PC")
        ]

        # First 10 PCs max
        pc_columns = pc_columns[:10]

        X = current[
            pc_columns
        ].to_numpy(
            dtype=float
        )

        # --------------------------------------------------------------
        # Geography
        # --------------------------------------------------------------

        geography_labels = (
            current["ActualClass"]
            .astype(str)
            .to_numpy()
        )

        if len(
            np.unique(
                geography_labels
            )
        ) >= 2:

            try:

                geography_score = float(
                    silhouette_score(
                        X,
                        geography_labels,
                    )
                )

            except Exception:
                geography_score = np.nan

        else:
            geography_score = np.nan

        # --------------------------------------------------------------
        # Year
        # --------------------------------------------------------------

        year_labels = (
            current["Year"]
            .astype(str)
            .to_numpy()
        )

        year_counts = pd.Series(
            year_labels
        ).value_counts()

        if (
            len(year_counts) >= 2
            and np.all(
                year_counts.values >= 2
            )
        ):

            try:

                year_score = float(
                    silhouette_score(
                        X,
                        year_labels,
                    )
                )

            except Exception:
                year_score = np.nan

        else:
            year_score = np.nan

        rows.append(
            {
                "Modality": modality,
                "GeographySilhouette": geography_score,
                "YearSilhouette": year_score,
                "Groups": int(
                    len(
                        np.unique(
                            geography_labels
                        )
                    )
                ),
                "Years": int(
                    len(
                        np.unique(
                            year_labels
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# GROUP × YEAR
# ==============================================================================

def build_group_year_table(
    mapping: pd.DataFrame,
    independent_ids: list[int],
) -> pd.DataFrame:

    current = mapping[
        mapping["SampleId"].isin(
            independent_ids
        )
    ].copy()

    return (
        current
        .groupby(
            [
                "ActualClass",
                "Year",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="Count"
        )
        .sort_values(
            [
                "ActualClass",
                "Year",
            ]
        )
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print("=" * 78)
    print(
        "HPLC METADATA / STRUCTURE DIAGNOSTICS"
    )
    print("=" * 78)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    mapping = load_mapping()

    independent_ids = (
        get_independent_sample_ids(
            mapping
        )
    )

    print(
        f"\nMapping rows       : {len(mapping)}"
    )

    print(
        f"Independent samples: "
        f"{len(independent_ids)}"
    )

    # ------------------------------------------------------------------
    # Load all wavelengths
    # ------------------------------------------------------------------

    raw_matrices = {}
    times = {}

    all_metrics = []

    for wavelength in WAVELENGTHS:

        print(
            f"\nLoading {wavelength} nm..."
        )

        matrix, time = load_sheet(
            wavelength
        )

        print(
            f"  Samples: {len(matrix)}"
        )

        print(
            f"  Points : {len(time)}"
        )

        print(
            f"  Time   : "
            f"{time.min():.6f} -> "
            f"{time.max():.6f}"
        )

        # Keep independent samples
        matrix = matrix.loc[
            matrix.index.isin(
                independent_ids
            )
        ].sort_index()

        raw_matrices[
            wavelength
        ] = matrix

        times[
            wavelength
        ] = time

        metrics = calculate_sample_metrics(
            matrix,
            time,
            wavelength,
        )

        all_metrics.append(
            metrics
        )

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Add mapping
    # ------------------------------------------------------------------

    sample_metrics = metrics_df.merge(
        mapping[
            [
                "SampleId",
                "ActualClass",
                "Year",
            ]
        ],
        on="SampleId",
        how="left",
    )

    # ------------------------------------------------------------------
    # Print scale summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("RAW SCALE SUMMARY")
    print("=" * 78)

    scale_summary = (
        sample_metrics
        .groupby("Wavelength")[
            [
                "AUC",
                "AbsAUC",
                "MaxIntensity",
                "MeanIntensity",
                "StdIntensity",
                "RangeIntensity",
            ]
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        scale_summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ------------------------------------------------------------------
    # Group summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("GROUP SCALE SUMMARY")
    print("=" * 78)

    group_summary = summarize_by_group(
        metrics_df,
        mapping,
    )

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Year summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("YEAR SCALE SUMMARY")
    print("=" * 78)

    year_summary = summarize_by_year(
        metrics_df,
        mapping,
    )

    print(
        year_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Group × Year
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("GROUP × YEAR")
    print("=" * 78)

    group_year = build_group_year_table(
        mapping,
        independent_ids,
    )

    print(
        group_year.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("PCA ANALYSIS")
    print("=" * 78)

    processed_matrices = {}

    for wavelength in WAVELENGTHS:

        processed_matrices[
            wavelength
        ] = prepare_processed_matrix(
            raw_matrices[wavelength],
            independent_ids,
        )

    modality_matrices = {
        "250 nm": processed_matrices[250],
        "308 nm": processed_matrices[308],
        "440 nm": processed_matrices[440],
        "Combined": np.hstack(
            [
                processed_matrices[250],
                processed_matrices[308],
                processed_matrices[440],
            ]
        ),
    }

    all_pca_scores = []
    all_pca_variance = []

    for modality, X in modality_matrices.items():

        print(
            f"\nPCA: {modality}"
        )

        score_df, variance_df = pca_analysis(
            X,
            independent_ids,
            mapping,
            modality,
        )

        all_pca_scores.append(
            score_df
        )

        all_pca_variance.append(
            variance_df
        )

        first_three = variance_df.head(3)

        print(
            first_three.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    pca_scores = pd.concat(
        all_pca_scores,
        ignore_index=True,
    )

    pca_variance = pd.concat(
        all_pca_variance,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # PCA structure
    # ------------------------------------------------------------------

    structure_summary = (
        pca_structure_summary(
            pca_scores
        )
    )

    print("\n")
    print("=" * 78)
    print("PCA VARIANCE STRUCTURE")
    print("=" * 78)

    print(
        pca_variance.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 78)
    print("PCA GEOGRAPHY vs YEAR")
    print("=" * 78)

    print(
        structure_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Silhouettes
    # ------------------------------------------------------------------

    silhouettes = (
        calculate_label_silhouettes(
            pca_scores
        )
    )

    print("\n")
    print("=" * 78)
    print("PCA SILHOUETTE: GEOGRAPHY vs YEAR")
    print("=" * 78)

    print(
        silhouettes.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Sample scale ratios by year
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("YEAR SCALE RATIO")
    print("=" * 78)

    year_scale_rows = []

    for wavelength in WAVELENGTHS:

        current = sample_metrics[
            sample_metrics[
                "Wavelength"
            ] == wavelength
        ].copy()

        year_groups = (
            current
            .groupby("Year")[
                [
                    "AbsAUC",
                    "MaxIntensity",
                    "MeanIntensity",
                ]
            ]
            .median()
        )

        years = list(
            year_groups.index
        )

        if len(years) >= 2:

            # Sort lexicographically/numerically
            years = sorted(
                years,
                key=lambda x: str(x)
            )

            y1 = years[0]
            y2 = years[-1]

            for metric in [
                "AbsAUC",
                "MaxIntensity",
                "MeanIntensity",
            ]:

                a = float(
                    year_groups.loc[
                        y1,
                        metric,
                    ]
                )

                b = float(
                    year_groups.loc[
                        y2,
                        metric,
                    ]
                )

                ratio = np.nan

                if abs(a) > 1e-12:
                    ratio = b / a

                year_scale_rows.append(
                    {
                        "Wavelength": wavelength,
                        "Metric": metric,
                        "YearA": y1,
                        "YearB": y2,
                        "MedianYearA": a,
                        "MedianYearB": b,
                        "YearB_over_YearA": ratio,
                    }
                )

    year_scale_ratio = pd.DataFrame(
        year_scale_rows
    )

    print(
        year_scale_ratio.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    sample_metrics_path = (
        OUTPUT_DIR
        / "metadata_structure_sample_metrics.csv"
    )

    group_summary_path = (
        OUTPUT_DIR
        / "metadata_structure_group_summary.csv"
    )

    year_summary_path = (
        OUTPUT_DIR
        / "metadata_structure_year_summary.csv"
    )

    group_year_path = (
        OUTPUT_DIR
        / "metadata_structure_group_year.csv"
    )

    pca_scores_path = (
        OUTPUT_DIR
        / "metadata_structure_pca_scores.csv"
    )

    pca_variance_path = (
        OUTPUT_DIR
        / "metadata_structure_pca_variance.csv"
    )

    pca_structure_path = (
        OUTPUT_DIR
        / "metadata_structure_pca_structure.csv"
    )

    silhouette_path = (
        OUTPUT_DIR
        / "metadata_structure_pca_silhouette.csv"
    )

    year_ratio_path = (
        OUTPUT_DIR
        / "metadata_structure_year_scale_ratio.csv"
    )

    sample_metrics.to_csv(
        sample_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_summary.to_csv(
        group_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    year_summary.to_csv(
        year_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_year.to_csv(
        group_year_path,
        index=False,
        encoding="utf-8-sig",
    )

    pca_scores.to_csv(
        pca_scores_path,
        index=False,
        encoding="utf-8-sig",
    )

    pca_variance.to_csv(
        pca_variance_path,
        index=False,
        encoding="utf-8-sig",
    )

    structure_summary.to_csv(
        pca_structure_path,
        index=False,
        encoding="utf-8-sig",
    )

    silhouettes.to_csv(
        silhouette_path,
        index=False,
        encoding="utf-8-sig",
    )

    year_scale_ratio.to_csv(
        year_ratio_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)

    print(
        f"Sample metrics : "
        f"{sample_metrics_path}"
    )

    print(
        f"Group summary  : "
        f"{group_summary_path}"
    )

    print(
        f"Year summary   : "
        f"{year_summary_path}"
    )

    print(
        f"Group × Year   : "
        f"{group_year_path}"
    )

    print(
        f"PCA scores     : "
        f"{pca_scores_path}"
    )

    print(
        f"PCA variance   : "
        f"{pca_variance_path}"
    )

    print(
        f"PCA structure  : "
        f"{pca_structure_path}"
    )

    print(
        f"PCA silhouette : "
        f"{silhouette_path}"
    )

    print(
        f"Year ratio     : "
        f"{year_ratio_path}"
    )


if __name__ == "__main__":
    main()