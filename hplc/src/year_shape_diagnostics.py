from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .metadata_structure_diagnostics import (
    load_mapping,
    load_sheet,
    asls_snv,
)


# ==============================================================================
# PATHS
# ==============================================================================

ROOT = Path(__file__).resolve().parents[1]

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

PCA_COMPONENTS = 10


# ==============================================================================
# HELPERS
# ==============================================================================

def independent_sample_ids(
    mapping: pd.DataFrame,
) -> list[int]:

    duplicate_second_ids = {
        b
        for _, b in DUPLICATE_PAIRS
    }

    ids = sorted(
        mapping["SampleId"]
        .astype(int)
        .unique()
        .tolist()
    )

    return [
        sid
        for sid in ids
        if sid not in duplicate_second_ids
    ]


def sample_metadata(
    mapping: pd.DataFrame,
    sample_ids: list[int],
) -> pd.DataFrame:

    return (
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
        .set_index("SampleId")
        .loc[sample_ids]
        .reset_index()
    )


def pearson_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return 0.0

    return float(
        np.corrcoef(a, b)[0, 1]
    )


# ==============================================================================
# NORMALIZATION METHODS
# ==============================================================================

def normalize_raw(
    y: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        y,
        dtype=float,
    )


def normalize_area(
    y: np.ndarray,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    area = np.sum(
        np.abs(y)
    )

    if area < 1e-12:
        return np.zeros_like(y)

    return y / area


def normalize_max(
    y: np.ndarray,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    scale = np.max(
        np.abs(y)
    )

    if scale < 1e-12:
        return np.zeros_like(y)

    return y / scale


def normalize_snv(
    y: np.ndarray,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    mean = np.mean(y)
    std = np.std(y)

    if std < 1e-12:
        return np.zeros_like(y)

    return (
        y - mean
    ) / std


def normalize_asls_snv(
    y: np.ndarray,
) -> np.ndarray:

    return asls_snv(
        np.asarray(
            y,
            dtype=float,
        )
    )


NORMALIZERS = {
    "Raw": normalize_raw,
    "AreaNormalized": normalize_area,
    "MaxNormalized": normalize_max,
    "SNV": normalize_snv,
    "AsLS+SNV": normalize_asls_snv,
}


# ==============================================================================
# LOAD RAW DATA
# ==============================================================================

def load_all_raw(
    mapping: pd.DataFrame,
    independent_ids: list[int],
) -> tuple[
    dict[int, pd.DataFrame],
    dict[int, np.ndarray],
]:

    matrices = {}
    times = {}

    for wavelength in WAVELENGTHS:

        print(
            f"Loading {wavelength} nm..."
        )

        matrix, time = load_sheet(
            wavelength
        )

        matrix = matrix.loc[
            matrix.index.isin(
                independent_ids
            )
        ].sort_index()

        matrices[wavelength] = matrix
        times[wavelength] = time

        print(
            f"  Samples: {len(matrix)}"
        )

        print(
            f"  Points : {len(time)}"
        )

    return matrices, times


# ==============================================================================
# BUILD NORMALIZED MODALITIES
# ==============================================================================

def build_normalized_matrices(
    raw_matrices: dict[int, pd.DataFrame],
    independent_ids: list[int],
) -> dict[str, dict[str, np.ndarray]]:

    result = {}

    for method_name, normalizer in NORMALIZERS.items():

        print(
            f"\nNormalization: {method_name}"
        )

        blocks = {}

        for wavelength in WAVELENGTHS:

            matrix = raw_matrices[
                wavelength
            ]

            processed = []

            for sid in independent_ids:

                y = matrix.loc[
                    sid
                ].to_numpy(
                    dtype=float
                )

                processed.append(
                    normalizer(y)
                )

            blocks[wavelength] = np.asarray(
                processed,
                dtype=float,
            )

        result[method_name] = {
            "250 nm": blocks[250],
            "308 nm": blocks[308],
            "440 nm": blocks[440],
            "Combined": np.hstack(
                [
                    blocks[250],
                    blocks[308],
                    blocks[440],
                ]
            ),
        }

    return result


# ==============================================================================
# PCA
# ==============================================================================

def run_pca(
    X: np.ndarray,
    metadata: pd.DataFrame,
    modality: str,
    normalization: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    n_components = min(
        PCA_COMPONENTS,
        X.shape[0] - 1,
        X.shape[1],
    )

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(
        X
    )

    score_df = metadata.copy()

    score_df["Normalization"] = (
        normalization
    )

    score_df["Modality"] = modality

    for i in range(
        n_components
    ):
        score_df[
            f"PC{i + 1}"
        ] = scores[:, i]

    variance_df = pd.DataFrame(
        {
            "Normalization": normalization,
            "Modality": modality,
            "PC": np.arange(
                1,
                n_components + 1,
            ),
            "ExplainedVariance": (
                pca.explained_variance_ratio_
            ),
        }
    )

    variance_df[
        "CumulativeVariance"
    ] = variance_df[
        "ExplainedVariance"
    ].cumsum()

    return (
        score_df,
        variance_df,
    )


# ==============================================================================
# YEAR SILHOUETTE
# ==============================================================================

def year_silhouette(
    score_df: pd.DataFrame,
) -> float:

    labels = (
        score_df["Year"]
        .astype(str)
        .to_numpy()
    )

    pc_columns = [
        c
        for c in score_df.columns
        if str(c).startswith("PC")
    ]

    X = score_df[
        pc_columns
    ].to_numpy(
        dtype=float
    )

    counts = (
        pd.Series(labels)
        .value_counts()
    )

    if (
        len(counts) < 2
        or np.any(
            counts.values < 2
        )
    ):
        return np.nan

    try:
        return float(
            silhouette_score(
                X,
                labels,
            )
        )

    except Exception:
        return np.nan


# ==============================================================================
# PCA YEAR STRUCTURE
# ==============================================================================

def calculate_eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=str,
    )

    grand_mean = np.mean(
        values
    )

    total_ss = np.sum(
        (
            values
            - grand_mean
        ) ** 2
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

        group_mean = np.mean(
            group_values
        )

        between_ss += (
            len(group_values)
            * (
                group_mean
                - grand_mean
            ) ** 2
        )

    return float(
        between_ss / total_ss
    )


# ==============================================================================
# YEAR CENTROID GEOMETRY
# ==============================================================================

def year_centroid_geometry(
    X: np.ndarray,
    metadata: pd.DataFrame,
    normalization: str,
    modality: str,
) -> dict:

    labels = (
        metadata["Year"]
        .astype(str)
        .to_numpy()
    )

    years = sorted(
        np.unique(labels)
    )

    if len(years) != 2:
        return {
            "YearA": np.nan,
            "YearB": np.nan,
            "CentroidSimilarity": np.nan,
            "CorrelationDistance": np.nan,
            "CentroidEuclideanDistance": np.nan,
        }

    centroids = {}

    for year in years:
        centroids[year] = np.mean(
            X[labels == year],
            axis=0,
        )

    a = centroids[years[0]]
    b = centroids[years[1]]

    similarity = pearson_similarity(
        a,
        b,
    )

    distance = float(
        1.0 - similarity
    )

    euclidean = float(
        np.linalg.norm(a - b)
    )

    return {
        "YearA": years[0],
        "YearB": years[1],
        "CentroidSimilarity": similarity,
        "CorrelationDistance": distance,
        "CentroidEuclideanDistance": euclidean,
    }


# ==============================================================================
# SAMPLE-LEVEL YEAR SEPARATION
# ==============================================================================

def sample_year_similarity(
    X: np.ndarray,
    metadata: pd.DataFrame,
    normalization: str,
    modality: str,
) -> pd.DataFrame:

    labels = (
        metadata["Year"]
        .astype(str)
        .to_numpy()
    )

    years = sorted(
        np.unique(labels)
    )

    if len(years) != 2:
        return pd.DataFrame()

    centroids = {}

    for year in years:
        centroids[year] = np.mean(
            X[labels == year],
            axis=0,
        )

    rows = []

    for idx in range(
        len(metadata)
    ):

        sample_id = int(
            metadata.loc[
                idx,
                "SampleId",
            ]
        )

        actual_year = str(
            metadata.loc[
                idx,
                "Year",
            ]
        )

        sample = X[idx]

        other_year = (
            years[1]
            if actual_year == years[0]
            else years[0]
        )

        own_similarity = pearson_similarity(
            sample,
            centroids[
                actual_year
            ],
        )

        other_similarity = pearson_similarity(
            sample,
            centroids[
                other_year
            ],
        )

        rows.append(
            {
                "SampleId": sample_id,
                "ActualYear": actual_year,
                "Normalization": normalization,
                "Modality": modality,
                "OwnYearSimilarity": own_similarity,
                "OtherYearSimilarity": other_similarity,
                "SimilarityGap": (
                    own_similarity
                    - other_similarity
                ),
                "OwnYearWin": (
                    own_similarity
                    >= other_similarity
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# SHARED GROUP / YEAR COMPARISON
# ==============================================================================

def common_groups_by_year(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        metadata
        .groupby(
            [
                "ActualClass",
                "Year",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    if len(counts.columns) != 2:
        return pd.DataFrame()

    year_a, year_b = (
        sorted(
            counts.columns.astype(str)
        )
    )

    counts.columns = [
        str(c)
        for c in counts.columns
    ]

    result = counts.copy()

    result["CommonInBothYears"] = (
        (result[str(year_a)] >= 2)
        & (result[str(year_b)] >= 2)
    )

    result = (
        result
        .reset_index()
        .rename(
            columns={
                "ActualClass": "Group"
            }
        )
    )

    result["YearA"] = year_a
    result["YearB"] = year_b

    return result


def shared_group_year_shape_comparison(
    X: np.ndarray,
    metadata: pd.DataFrame,
    normalization: str,
    modality: str,
) -> pd.DataFrame:

    years = sorted(
        metadata["Year"]
        .astype(str)
        .unique()
    )

    if len(years) != 2:
        return pd.DataFrame()

    year_a, year_b = years

    rows = []

    groups = sorted(
        metadata[
            "ActualClass"
        ]
        .astype(str)
        .unique()
    )

    for group in groups:

        mask_a = (
            metadata["ActualClass"]
            .astype(str)
            .eq(str(group))
            & metadata["Year"]
            .astype(str)
            .eq(year_a)
        )

        mask_b = (
            metadata["ActualClass"]
            .astype(str)
            .eq(str(group))
            & metadata["Year"]
            .astype(str)
            .eq(year_b)
        )

        n_a = int(
            mask_a.sum()
        )

        n_b = int(
            mask_b.sum()
        )

        if (
            n_a < 2
            or n_b < 2
        ):
            continue

        centroid_a = np.mean(
            X[mask_a.to_numpy()],
            axis=0,
        )

        centroid_b = np.mean(
            X[mask_b.to_numpy()],
            axis=0,
        )

        similarity = pearson_similarity(
            centroid_a,
            centroid_b,
        )

        rows.append(
            {
                "Normalization": normalization,
                "Modality": modality,
                "Group": group,
                "YearA": year_a,
                "YearB": year_b,
                "SamplesYearA": n_a,
                "SamplesYearB": n_b,
                "WithinGroupCrossYearSimilarity": similarity,
                "CorrelationDistance": (
                    1.0 - similarity
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# RAW SCALE SUMMARY
# ==============================================================================

def raw_scale_summary(
    raw_matrices: dict[int, pd.DataFrame],
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for wavelength in WAVELENGTHS:

        matrix = raw_matrices[
            wavelength
        ]

        for sid in matrix.index:

            y = matrix.loc[
                sid
            ].to_numpy(
                dtype=float
            )

            rows.append(
                {
                    "SampleId": int(sid),
                    "Wavelength": wavelength,
                    "AbsAUC": float(
                        np.sum(
                            np.abs(y)
                        )
                    ),
                    "MaxAbs": float(
                        np.max(
                            np.abs(y)
                        )
                    ),
                    "MeanAbs": float(
                        np.mean(
                            np.abs(y)
                        )
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    ).merge(
        metadata,
        on="SampleId",
        how="left",
    )

    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print("=" * 78)
    print(
        "HPLC YEAR / SHAPE DIAGNOSTICS"
    )
    print("=" * 78)

    mapping = load_mapping()

    ids = independent_sample_ids(
        mapping
    )

    metadata = sample_metadata(
        mapping,
        ids,
    )

    print(
        f"\nIndependent samples: {len(ids)}"
    )

    print(
        "Years:",
        ", ".join(
            sorted(
                metadata["Year"]
                .astype(str)
                .unique()
            )
        ),
    )

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    raw_matrices, times = load_all_raw(
        mapping,
        ids,
    )

    # ------------------------------------------------------------------
    # Raw scale summary
    # ------------------------------------------------------------------

    scale_df = raw_scale_summary(
        raw_matrices,
        metadata,
    )

    print("\n")
    print("=" * 78)
    print("RAW YEAR SCALE")
    print("=" * 78)

    raw_year_summary = (
        scale_df
        .groupby(
            [
                "Wavelength",
                "Year",
            ]
        )[
            [
                "AbsAUC",
                "MaxAbs",
                "MeanAbs",
            ]
        ]
        .median()
        .reset_index()
    )

    print(
        raw_year_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ------------------------------------------------------------------
    # Common groups
    # ------------------------------------------------------------------

    shared_groups = (
        common_groups_by_year(
            metadata
        )
    )

    print("\n")
    print("=" * 78)
    print("GROUPS WITH >=2 SAMPLES IN BOTH YEARS")
    print("=" * 78)

    if shared_groups.empty:
        print(
            "No groups have >=2 independent samples "
            "in both years."
        )
    else:

        print(
            shared_groups[
                shared_groups[
                    "CommonInBothYears"
                ]
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Build all normalized modalities
    # ------------------------------------------------------------------

    normalized = (
        build_normalized_matrices(
            raw_matrices,
            ids,
        )
    )

    all_scores = []
    all_variance = []
    all_summary = []
    all_pca_structure = []
    all_sample_year = []
    all_shared_groups = []

    # ------------------------------------------------------------------
    # Analyze every normalization / modality
    # ------------------------------------------------------------------

    for normalization, modalities in (
        normalized.items()
    ):

        for modality, X in modalities.items():

            print("\n")
            print(
                "-" * 78
            )

            print(
                f"{normalization} | {modality}"
            )

            print(
                "-" * 78
            )

            # ----------------------------------------------------------
            # PCA
            # ----------------------------------------------------------

            scores, variance = run_pca(
                X,
                metadata,
                modality,
                normalization,
            )

            all_scores.append(
                scores
            )

            all_variance.append(
                variance
            )

            # ----------------------------------------------------------
            # Year silhouette
            # ----------------------------------------------------------

            silhouette = year_silhouette(
                scores
            )

            # ----------------------------------------------------------
            # Year centroid geometry
            # ----------------------------------------------------------

            geometry = (
                year_centroid_geometry(
                    X,
                    metadata,
                    normalization,
                    modality,
                )
            )

            # ----------------------------------------------------------
            # PC Year eta squared
            # ----------------------------------------------------------

            labels = (
                metadata["Year"]
                .astype(str)
                .to_numpy()
            )

            pc_rows = []

            pc_columns = [
                c
                for c in scores.columns
                if str(c).startswith("PC")
            ]

            for pc in pc_columns:

                values = scores[
                    pc
                ].to_numpy(
                    dtype=float
                )

                pc_rows.append(
                    {
                        "Normalization": normalization,
                        "Modality": modality,
                        "PC": pc,
                        "YearEtaSquared": (
                            calculate_eta_squared(
                                values,
                                labels,
                            )
                        ),
                    }
                )

            all_pca_structure.append(
                pd.DataFrame(
                    pc_rows
                )
            )

            # ----------------------------------------------------------
            # Summary
            # ----------------------------------------------------------

            row = {
                "Normalization": normalization,
                "Modality": modality,
                "Samples": len(
                    metadata
                ),
                "Years": 2,
                "YearSilhouette": silhouette,
                **geometry,
            }

            all_summary.append(
                row
            )

            print(
                f"Year silhouette: "
                f"{silhouette:.4f}"
                if not np.isnan(
                    silhouette
                )
                else
                "Year silhouette: NA"
            )

            print(
                "Centroid similarity: "
                f"{geometry['CentroidSimilarity']:.4f}"
            )

            print(
                "Correlation distance: "
                f"{geometry['CorrelationDistance']:.4f}"
            )

            print(
                "Centroid Euclidean distance: "
                f"{geometry['CentroidEuclideanDistance']:.4f}"
            )

            # ----------------------------------------------------------
            # Sample-level year similarity
            # ----------------------------------------------------------

            sample_year = (
                sample_year_similarity(
                    X,
                    metadata,
                    normalization,
                    modality,
                )
            )

            all_sample_year.append(
                sample_year
            )

            # ----------------------------------------------------------
            # Shared group comparison
            # ----------------------------------------------------------

            shared = (
                shared_group_year_shape_comparison(
                    X,
                    metadata,
                    normalization,
                    modality,
                )
            )

            if not shared.empty:
                all_shared_groups.append(
                    shared
                )

            # ----------------------------------------------------------
            # Show PCA first 3 PCs
            # ----------------------------------------------------------

            print(
                "\nPCA first 3 PCs:"
            )

            print(
                variance.head(3).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------

    summary_df = pd.DataFrame(
        all_summary
    )

    pca_scores_df = pd.concat(
        all_scores,
        ignore_index=True,
    )

    pca_variance_df = pd.concat(
        all_variance,
        ignore_index=True,
    )

    pca_structure_df = pd.concat(
        all_pca_structure,
        ignore_index=True,
    )

    sample_year_df = pd.concat(
        all_sample_year,
        ignore_index=True,
    )

    if all_shared_groups:
        shared_group_df = pd.concat(
            all_shared_groups,
            ignore_index=True,
        )
    else:
        shared_group_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # YEAR SUMMARY
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("YEAR SHAPE SUMMARY")
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # BEST YEAR SEPARATION PER NORMALIZATION
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("YEAR SILHOUETTE BY NORMALIZATION")
    print("=" * 78)

    print(
        summary_df[
            [
                "Normalization",
                "Modality",
                "YearSilhouette",
                "CentroidSimilarity",
                "CorrelationDistance",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # SHARED GROUP SHAPE
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "COMMON-GROUP CROSS-YEAR SHAPE SIMILARITY"
    )
    print("=" * 78)

    if shared_group_df.empty:
        print(
            "No groups have >=2 samples in both years."
        )
    else:

        print(
            shared_group_df.sort_values(
                [
                    "Normalization",
                    "Modality",
                    "WithinGroupCrossYearSimilarity",
                ]
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # SAMPLES MOST YEAR-AMBIGUOUS
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "MOST YEAR-AMBIGUOUS SAMPLES"
    )
    print("=" * 78)

    ambiguous = (
        sample_year_df
        .assign(
            AbsGap=lambda df:
                np.abs(
                    df["SimilarityGap"]
                )
        )
        .sort_values(
            "AbsGap",
            ascending=True,
        )
        .head(40)
    )

    print(
        ambiguous[
            [
                "SampleId",
                "ActualYear",
                "Normalization",
                "Modality",
                "OwnYearSimilarity",
                "OtherYearSimilarity",
                "SimilarityGap",
                "OwnYearWin",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # PCA YEAR ETA²
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "TOP PCA YEAR Eta²"
    )
    print("=" * 78)

    top_eta = (
        pca_structure_df
        .sort_values(
            "YearEtaSquared",
            ascending=False,
        )
        .head(40)
    )

    print(
        top_eta.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    paths = {
        "raw_scale":
            OUTPUT_DIR
            / "year_shape_raw_scale.csv",

        "year_summary":
            OUTPUT_DIR
            / "year_shape_summary.csv",

        "pca_scores":
            OUTPUT_DIR
            / "year_shape_pca_scores.csv",

        "pca_variance":
            OUTPUT_DIR
            / "year_shape_pca_variance.csv",

        "pca_structure":
            OUTPUT_DIR
            / "year_shape_pca_structure.csv",

        "sample_year":
            OUTPUT_DIR
            / "year_shape_sample_similarity.csv",

        "shared_groups":
            OUTPUT_DIR
            / "year_shape_common_groups.csv",
    }

    scale_df.to_csv(
        paths["raw_scale"],
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        paths["year_summary"],
        index=False,
        encoding="utf-8-sig",
    )

    pca_scores_df.to_csv(
        paths["pca_scores"],
        index=False,
        encoding="utf-8-sig",
    )

    pca_variance_df.to_csv(
        paths["pca_variance"],
        index=False,
        encoding="utf-8-sig",
    )

    pca_structure_df.to_csv(
        paths["pca_structure"],
        index=False,
        encoding="utf-8-sig",
    )

    sample_year_df.to_csv(
        paths["sample_year"],
        index=False,
        encoding="utf-8-sig",
    )

    shared_group_df.to_csv(
        paths["shared_groups"],
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)

    for name, path in paths.items():
        print(
            f"{name:18s}: {path}"
        )


if __name__ == "__main__":
    main()