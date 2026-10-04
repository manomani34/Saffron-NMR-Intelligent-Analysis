from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score


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


# ==============================================================================
# ALS BASELINE
# ==============================================================================

def baseline_asls(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """
    Asymmetric least-squares baseline correction.

    No learned parameters are shared between samples.
    """

    y = np.asarray(y, dtype=float)

    n = len(y)

    if n < 3:
        return np.zeros_like(y)

    # Second-order difference matrix
    d = np.diff(np.eye(n), 2, axis=0)

    # D^T D
    dtd = d.T @ d

    weights = np.ones(n)

    for _ in range(n_iter):
        w = np.diag(weights)
        z = np.linalg.solve(w + lam * dtd, weights * y)

        weights = np.where(
            y > z,
            p,
            1.0 - p,
        )

    return z


def asls_snv(y: np.ndarray) -> np.ndarray:
    """
    AsLS baseline correction followed by SNV.
    """

    y = np.asarray(y, dtype=float)

    # Baseline correction
    baseline = baseline_asls(y)

    corrected = y - baseline

    # SNV
    mean = np.mean(corrected)
    std = np.std(corrected)

    if std < 1e-12:
        return np.zeros_like(corrected)

    return (corrected - mean) / std


# ==============================================================================
# EXCEL LOADING
# ==============================================================================

def find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(len(df), 30)):
        values = [
            str(x).strip().lower()
            for x in df.iloc[i].tolist()
        ]

        if "number" in values and "group" in values:
            return i

    raise ValueError("Could not find HPLC header row.")


def load_sheet(
    wavelength: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sheet_name = f"{wavelength}"

    # Sheet names in the client workbook
    if wavelength == 440:
        sheet_name = "Crocin 440"
    elif wavelength == 250:
        sheet_name = "picrocrocin 250"
    elif wavelength == 308:
        sheet_name = "safranal 308"

    raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name=sheet_name,
        header=None,
    )

    header_row = find_header_row(raw)

    header = raw.iloc[header_row].copy()

    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header

    # Find Sample Number column
    number_col = None
    group_col = None

    for col in data.columns:
        text = str(col).strip().lower()

        if text == "number":
            number_col = col

        if text == "group":
            group_col = col

    if number_col is None:
        raise ValueError(f"Number column not found in {sheet_name}")

    # Numeric time columns only, restricted to 0..30 min
    time_columns = []

    for col in data.columns:
        try:
            value = float(col)

            if 0 < value <= MAX_TIME:
                time_columns.append(col)

        except Exception:
            continue

    if not time_columns:
        raise ValueError(f"No time columns found in {sheet_name}")

    # Sort by actual numeric time
    time_columns = sorted(
        time_columns,
        key=lambda x: float(x),
    )

    sample_ids = pd.to_numeric(
        data[number_col],
        errors="coerce",
    )

    valid_rows = sample_ids.notna()

    data = data.loc[valid_rows].copy()
    sample_ids = sample_ids.loc[valid_rows].astype(int).to_numpy()

    matrix = data[time_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    # Fill missing values along time
    matrix_df = pd.DataFrame(matrix)

    matrix_df = (
        matrix_df
        .interpolate(
            axis=1,
            limit_direction="both",
        )
        .fillna(0.0)
    )

    matrix = matrix_df.to_numpy(dtype=float)

    return (
        pd.DataFrame(
            matrix,
            index=sample_ids,
            columns=[float(x) for x in time_columns],
        ),
        sample_ids,
        np.array([float(x) for x in time_columns]),
    )


# ==============================================================================
# MAPPING
# ==============================================================================

def load_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING_PATH)

    # Normalize column names
    mapping.columns = [
        str(c).strip()
        for c in mapping.columns
    ]

    # Detect sample id column
    sample_col = None
    group_col = None
    year_col = None

    for col in mapping.columns:
        c = col.lower()

        if c in {"sampleid", "sample_id", "number", "sample"}:
            sample_col = col

        if c in {"group", "class", "region"}:
            group_col = col

        if c in {"year", "harvestyear", "harvest_year"}:
            year_col = col

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

    result["SampleId"] = result["SampleId"].astype(int)

    return result


# ==============================================================================
# INDEPENDENT SAMPLES
# ==============================================================================

def get_independent_sample_ids(
    mapping: pd.DataFrame,
) -> list[int]:
    sample_ids = sorted(
        mapping["SampleId"]
        .astype(int)
        .unique()
        .tolist()
    )

    duplicate_second_ids = {
        b
        for _, b in DUPLICATE_PAIRS
    }

    independent = [
        sid
        for sid in sample_ids
        if sid not in duplicate_second_ids
    ]

    return independent


# ==============================================================================
# BUILD MATRICES
# ==============================================================================

def build_processed_matrices(
    mapping: pd.DataFrame,
) -> tuple[
    dict[int, pd.DataFrame],
    dict[str, np.ndarray],
    pd.DataFrame,
]:

    matrices: dict[int, pd.DataFrame] = {}
    times: dict[str, np.ndarray] = {}

    for wl in WAVELENGTHS:

        print(f"Loading {wl} nm...")

        matrix_df, sample_ids, time = load_sheet(wl)

        print(f"  Samples: {len(sample_ids)}")
        print(f"  Points : {len(time)}")
        print(
            f"  Time   : "
            f"{time.min():.6f} -> "
            f"{time.max():.6f}"
        )

        # Keep only mapped + independent samples
        matrix_df = matrix_df.loc[
            matrix_df.index.isin(
                get_independent_sample_ids(mapping)
            )
        ]

        # Ensure sample order is stable
        matrix_df = matrix_df.sort_index()

        processed = []

        for _, row in matrix_df.iterrows():
            processed.append(
                asls_snv(row.to_numpy(dtype=float))
            )

        processed_df = pd.DataFrame(
            processed,
            index=matrix_df.index,
            columns=matrix_df.columns,
        )

        matrices[wl] = processed_df
        times[str(wl)] = time

    return matrices, times, mapping


# ==============================================================================
# LABEL ALIGNMENT
# ==============================================================================

def align_samples(
    matrices: dict[int, pd.DataFrame],
    mapping: pd.DataFrame,
) -> tuple[
    list[int],
    np.ndarray,
    dict[int, np.ndarray],
]:

    common_ids = set(
        mapping["SampleId"].astype(int)
    )

    for matrix in matrices.values():
        common_ids &= set(
            matrix.index.astype(int)
        )

    sample_ids = sorted(common_ids)

    label_map = (
        mapping
        .set_index("SampleId")["ActualClass"]
        .to_dict()
    )

    labels = np.array(
        [label_map[sid] for sid in sample_ids]
    )

    aligned = {}

    for wl, matrix in matrices.items():
        aligned[wl] = (
            matrix
            .loc[sample_ids]
            .to_numpy(dtype=float)
        )

    return sample_ids, labels, aligned


# ==============================================================================
# SILHOUETTE
# ==============================================================================

def compute_silhouette_for_modality(
    X: np.ndarray,
    labels: np.ndarray,
    modality: str,
    sample_ids: list[int],
) -> tuple[pd.DataFrame, dict]:

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError(
            f"{modality}: at least two groups are required."
        )

    # Every class currently has >= 2 independent samples.
    if any(
        np.sum(labels == g) < 2
        for g in unique_labels
    ):
        raise ValueError(
            f"{modality}: silhouette requires "
            f"at least 2 samples per group."
        )

    print(
        f"\nCalculating silhouette: {modality}"
    )

    # ------------------------------------------------------------------
    # Correlation-distance silhouette
    # ------------------------------------------------------------------

    corr_values = silhouette_samples(
        X,
        labels,
        metric="correlation",
    )

    corr_score = float(
        silhouette_score(
            X,
            labels,
            metric="correlation",
        )
    )

    # ------------------------------------------------------------------
    # Euclidean silhouette
    # ------------------------------------------------------------------

    euclidean_values = silhouette_samples(
        X,
        labels,
        metric="euclidean",
    )

    euclidean_score = float(
        silhouette_score(
            X,
            labels,
            metric="euclidean",
        )
    )

    rows = []

    for i, sid in enumerate(sample_ids):

        rows.append(
            {
                "SampleId": sid,
                "ActualClass": labels[i],
                "Modality": modality,
                "SilhouetteCorrelation": corr_values[i],
                "SilhouetteEuclidean": euclidean_values[i],
                "CorrelationStructure": (
                    "OwnCluster"
                    if corr_values[i] >= 0
                    else "OutsideOwnCluster"
                ),
                "EuclideanStructure": (
                    "OwnCluster"
                    if euclidean_values[i] >= 0
                    else "OutsideOwnCluster"
                ),
            }
        )

    sample_df = pd.DataFrame(rows)

    group_rows = []

    for group in sorted(unique_labels):

        mask = labels == group

        group_corr = corr_values[mask]
        group_euc = euclidean_values[mask]

        group_rows.append(
            {
                "Modality": modality,
                "Group": group,
                "Samples": int(mask.sum()),
                "MeanSilhouetteCorrelation": (
                    float(np.mean(group_corr))
                ),
                "MedianSilhouetteCorrelation": (
                    float(np.median(group_corr))
                ),
                "MinSilhouetteCorrelation": (
                    float(np.min(group_corr))
                ),
                "MaxSilhouetteCorrelation": (
                    float(np.max(group_corr))
                ),
                "PositiveCorrelationFraction": (
                    float(np.mean(group_corr >= 0))
                ),
                "MeanSilhouetteEuclidean": (
                    float(np.mean(group_euc))
                ),
                "MedianSilhouetteEuclidean": (
                    float(np.median(group_euc))
                ),
                "PositiveEuclideanFraction": (
                    float(np.mean(group_euc >= 0))
                ),
            }
        )

    group_df = pd.DataFrame(group_rows)

    summary = {
        "Modality": modality,
        "Samples": len(sample_ids),
        "Groups": len(unique_labels),

        "SilhouetteCorrelation": corr_score,
        "SilhouetteEuclidean": euclidean_score,

        "MeanSampleSilhouetteCorrelation": float(
            np.mean(corr_values)
        ),

        "MedianSampleSilhouetteCorrelation": float(
            np.median(corr_values)
        ),

        "NegativeCorrelationFraction": float(
            np.mean(corr_values < 0)
        ),

        "PositiveCorrelationFraction": float(
            np.mean(corr_values >= 0)
        ),

        "MeanSampleSilhouetteEuclidean": float(
            np.mean(euclidean_values)
        ),

        "MedianSampleSilhouetteEuclidean": float(
            np.median(euclidean_values)
        ),

        "NegativeEuclideanFraction": float(
            np.mean(euclidean_values < 0)
        ),

        "PositiveEuclideanFraction": float(
            np.mean(euclidean_values >= 0)
        ),
    }

    return (
        sample_df,
        {
            "group_df": group_df,
            "summary": summary,
        },
    )


# ==============================================================================
# GROUP PAIR DISTANCES
# ==============================================================================

def group_pair_correlation_similarity(
    X: np.ndarray,
    labels: np.ndarray,
    modality: str,
) -> pd.DataFrame:

    groups = sorted(np.unique(labels))

    centroids = {}

    for group in groups:
        centroids[group] = np.mean(
            X[labels == group],
            axis=0,
        )

    rows = []

    for g1, g2 in combinations(groups, 2):

        a = centroids[g1]
        b = centroids[g2]

        a_std = np.std(a)
        b_std = np.std(b)

        if a_std < 1e-12 or b_std < 1e-12:
            corr = 0.0
        else:
            corr = float(
                np.corrcoef(a, b)[0, 1]
            )

        distance = 1.0 - corr

        rows.append(
            {
                "Modality": modality,
                "GroupA": g1,
                "GroupB": g2,
                "CentroidCorrelationSimilarity": corr,
                "CorrelationDistance": distance,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "CentroidCorrelationSimilarity",
        ascending=False,
    )


# ==============================================================================
# PRINT
# ==============================================================================

def print_results(
    summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    sample_df: pd.DataFrame,
):

    print("\n")
    print("=" * 78)
    print("SILHOUETTE SUMMARY")
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 78)
    print("GROUP SILHOUETTE")
    print("=" * 78)

    print(
        group_df[
            [
                "Modality",
                "Group",
                "Samples",
                "MeanSilhouetteCorrelation",
                "PositiveCorrelationFraction",
                "MeanSilhouetteEuclidean",
                "PositiveEuclideanFraction",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 78)
    print("MOST NEGATIVE SAMPLE SILHOUETTES")
    print("=" * 78)

    negative = (
        sample_df[
            [
                "SampleId",
                "ActualClass",
                "Modality",
                "SilhouetteCorrelation",
                "SilhouetteEuclidean",
            ]
        ]
        .sort_values(
            "SilhouetteCorrelation",
            ascending=True,
        )
        .head(40)
    )

    print(
        negative.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print("=" * 78)
    print("HPLC SILHOUETTE ANALYSIS")
    print("=" * 78)

    mapping = load_mapping()

    print(f"\nMapping rows: {len(mapping)}")

    independent_ids = get_independent_sample_ids(
        mapping
    )

    print(
        f"Independent samples: "
        f"{len(independent_ids)}"
    )

    matrices, _, mapping = build_processed_matrices(
        mapping
    )

    sample_ids, labels, aligned = align_samples(
        matrices,
        mapping,
    )

    print(
        f"\nCommon independent samples: "
        f"{len(sample_ids)}"
    )

    print(
        f"Groups: "
        f"{len(np.unique(labels))}"
    )

    # ------------------------------------------------------------------
    # Modality matrices
    # ------------------------------------------------------------------

    modality_matrices = {
        "250 nm": aligned[250],
        "308 nm": aligned[308],
        "440 nm": aligned[440],
    }

    modality_matrices["Combined"] = np.hstack(
        [
            aligned[250],
            aligned[308],
            aligned[440],
        ]
    )

    # ------------------------------------------------------------------
    # Silhouette
    # ------------------------------------------------------------------

    all_samples = []
    all_groups = []
    summaries = []
    overlaps = []

    for modality, X in modality_matrices.items():

        sample_df, details = (
            compute_silhouette_for_modality(
                X,
                labels,
                modality,
                sample_ids,
            )
        )

        all_samples.append(sample_df)
        all_groups.append(details["group_df"])
        summaries.append(details["summary"])

        overlaps.append(
            group_pair_correlation_similarity(
                X,
                labels,
                modality,
            )
        )

    sample_df = pd.concat(
        all_samples,
        ignore_index=True,
    )

    group_df = pd.concat(
        all_groups,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    overlaps_df = pd.concat(
        overlaps,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Extra negative-silhouette table
    # ------------------------------------------------------------------

    negative_df = (
        sample_df[
            sample_df["SilhouetteCorrelation"] < 0
        ]
        .sort_values(
            ["Modality", "SilhouetteCorrelation"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    summary_path = (
        OUTPUT_DIR /
        "silhouette_summary.csv"
    )

    group_path = (
        OUTPUT_DIR /
        "silhouette_group_performance.csv"
    )

    sample_path = (
        OUTPUT_DIR /
        "silhouette_samples.csv"
    )

    negative_path = (
        OUTPUT_DIR /
        "silhouette_negative_samples.csv"
    )

    overlap_path = (
        OUTPUT_DIR /
        "silhouette_group_overlaps.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_df.to_csv(
        group_path,
        index=False,
        encoding="utf-8-sig",
    )

    sample_df.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    negative_df.to_csv(
        negative_path,
        index=False,
        encoding="utf-8-sig",
    )

    overlaps_df.to_csv(
        overlap_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    print_results(
        summary_df,
        group_df,
        sample_df,
    )

    print("\n")
    print("=" * 78)
    print("TOP GROUP OVERLAPS")
    print("=" * 78)

    print(
        overlaps_df.head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)

    print(f"Summary       : {summary_path}")
    print(f"Group         : {group_path}")
    print(f"Samples       : {sample_path}")
    print(f"Negative      : {negative_path}")
    print(f"Group overlaps: {overlap_path}")


if __name__ == "__main__":
    main()