from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ==============================================================================
# IMPORT EXISTING HPLC FUNCTIONS
# ==============================================================================

from .metadata_structure_diagnostics import (
    load_mapping,
    get_independent_sample_ids,
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

MIN_GROUP_SIZE_FOR_STRUCTURE = 2

PCA_COMPONENTS = 10


# ==============================================================================
# INDEPENDENT IDS
# ==============================================================================

def get_independent_ids(
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


# ==============================================================================
# LOAD + PROCESS
# ==============================================================================

def load_processed_matrices(
    mapping: pd.DataFrame,
) -> dict[int, np.ndarray]:

    independent_ids = get_independent_ids(
        mapping
    )

    matrices = {}

    for wavelength in WAVELENGTHS:

        print(
            f"Loading {wavelength} nm..."
        )

        raw_matrix, time = load_sheet(
            wavelength
        )

        raw_matrix = raw_matrix.loc[
            raw_matrix.index.isin(
                independent_ids
            )
        ].sort_index()

        processed = []

        for sample_id in independent_ids:

            y = raw_matrix.loc[
                sample_id
            ].to_numpy(
                dtype=float
            )

            processed.append(
                asls_snv(y)
            )

        matrices[wavelength] = np.asarray(
            processed,
            dtype=float
        )

    return matrices


# ==============================================================================
# BUILD MODALITIES
# ==============================================================================

def build_modalities(
    processed: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:

    return {
        "250 nm": processed[250],
        "308 nm": processed[308],
        "440 nm": processed[440],
        "Combined": np.hstack(
            [
                processed[250],
                processed[308],
                processed[440],
            ]
        ),
    }


# ==============================================================================
# MAPPING TABLE
# ==============================================================================

def build_sample_metadata(
    mapping: pd.DataFrame,
    sample_ids: list[int],
) -> pd.DataFrame:

    meta = (
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

    return meta


# ==============================================================================
# PCA WITHIN YEAR
# ==============================================================================

def pca_within_year(
    X: np.ndarray,
    metadata: pd.DataFrame,
    modality: str,
    year,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    mask = (
        metadata["Year"].astype(str)
        == str(year)
    )

    meta_year = (
        metadata.loc[mask]
        .reset_index(drop=True)
    )

    X_year = X[mask.to_numpy()]

    if len(meta_year) < 3:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    n_components = min(
        PCA_COMPONENTS,
        X_year.shape[0] - 1,
        X_year.shape[1],
    )

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(
        X_year
    )

    score_df = meta_year.copy()

    score_df["Modality"] = modality

    for i in range(
        n_components
    ):
        score_df[
            f"PC{i + 1}"
        ] = scores[:, i]

    variance_df = pd.DataFrame(
        {
            "Modality": modality,
            "Year": str(year),
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

    labels = np.asarray(
        labels,
        dtype=str,
    )

    overall_mean = np.mean(
        values
    )

    total_ss = np.sum(
        (
            values
            - overall_mean
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
# PCA GEOGRAPHY STRUCTURE
# ==============================================================================

def calculate_pca_structure(
    score_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        modality,
        year,
    ), current in score_df.groupby(
        [
            "Modality",
            "Year",
        ]
    ):

        pc_columns = [
            c
            for c in current.columns
            if str(c).startswith("PC")
        ]

        for pc in pc_columns:

            values = current[
                pc
            ].to_numpy(
                dtype=float
            )

            labels = (
                current["ActualClass"]
                .astype(str)
                .to_numpy()
            )

            rows.append(
                {
                    "Modality": modality,
                    "Year": str(year),
                    "PC": pc,
                    "GeographyEtaSquared": (
                        eta_squared(
                            values,
                            labels,
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# ELIGIBLE GROUPS
# ==============================================================================

def eligible_groups(
    labels: np.ndarray,
) -> list[str]:

    counts = (
        pd.Series(labels)
        .value_counts()
    )

    return sorted(
        counts[
            counts >= MIN_GROUP_SIZE_FOR_STRUCTURE
        ].index
        .astype(str)
        .tolist()
    )


# ==============================================================================
# GEOGRAPHY SILHOUETTE
# ==============================================================================

def calculate_silhouette(
    score_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        modality,
        year,
    ), current in score_df.groupby(
        [
            "Modality",
            "Year",
        ]
    ):

        labels_all = (
            current["ActualClass"]
            .astype(str)
            .to_numpy()
        )

        groups = eligible_groups(
            labels_all
        )

        # Keep only groups with >= 2 samples.
        mask = np.isin(
            labels_all,
            groups,
        )

        current_eligible = (
            current.loc[mask]
            .reset_index(drop=True)
        )

        if (
            len(current_eligible)
            < 3
        ):
            continue

        labels = (
            current_eligible[
                "ActualClass"
            ]
            .astype(str)
            .to_numpy()
        )

        pc_columns = [
            c
            for c in current_eligible.columns
            if str(c).startswith("PC")
        ]

        X = current_eligible[
            pc_columns
        ].to_numpy(
            dtype=float
        )

        if len(
            np.unique(labels)
        ) < 2:
            continue

        try:
            silhouette = float(
                silhouette_score(
                    X,
                    labels,
                )
            )

        except Exception:
            silhouette = np.nan

        rows.append(
            {
                "Modality": modality,
                "Year": str(year),
                "EligibleGroups": len(groups),
                "EligibleSamples": len(
                    current_eligible
                ),
                "GeographySilhouette": silhouette,
            }
        )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# CENTROID STRUCTURE
# ==============================================================================

def centroid_structure(
    X: np.ndarray,
    metadata: pd.DataFrame,
    sample_ids: list[int],
    modality: str,
    year,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    mask = (
        metadata["Year"].astype(str)
        == str(year)
    )

    meta_year = (
        metadata.loc[mask]
        .reset_index(drop=True)
    )

    X_year = X[
        mask.to_numpy()
    ]

    labels = (
        meta_year[
            "ActualClass"
        ]
        .astype(str)
        .to_numpy()
    )

    groups = eligible_groups(
        labels
    )

    # Keep only groups with >=2 samples
    eligible_mask = np.isin(
        labels,
        groups,
    )

    labels = labels[
        eligible_mask
    ]

    X_year = X_year[
        eligible_mask
    ]

    meta_year = meta_year.loc[
        eligible_mask
    ].reset_index(drop=True)

    if len(groups) < 2:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    centroids = {}

    group_sizes = {}

    for group in groups:

        group_mask = (
            labels == group
        )

        centroids[group] = np.mean(
            X_year[group_mask],
            axis=0,
        )

        group_sizes[group] = int(
            group_mask.sum()
        )

    # --------------------------------------------------------------
    # Pairwise centroid overlap
    # --------------------------------------------------------------

    pair_rows = []

    for group_a, group_b in combinations(
        groups,
        2,
    ):

        a = centroids[group_a]
        b = centroids[group_b]

        a_std = np.std(a)
        b_std = np.std(b)

        if (
            a_std < 1e-12
            or b_std < 1e-12
        ):
            similarity = 0.0

        else:
            similarity = float(
                np.corrcoef(
                    a,
                    b,
                )[0, 1]
            )

        distance = float(
            1.0 - similarity
        )

        pair_rows.append(
            {
                "Modality": modality,
                "Year": str(year),
                "GroupA": group_a,
                "GroupB": group_b,
                "GroupASize": group_sizes[group_a],
                "GroupBSize": group_sizes[group_b],
                "CentroidCorrelationSimilarity": (
                    similarity
                ),
                "CorrelationDistance": distance,
            }
        )

    pair_df = pd.DataFrame(
        pair_rows
    ).sort_values(
        "CentroidCorrelationSimilarity",
        ascending=False,
    )

    # --------------------------------------------------------------
    # Nearest-centroid result per sample
    # --------------------------------------------------------------

    sample_rows = []

    for idx in range(
        len(labels)
    ):

        sid = int(
            meta_year.loc[
                idx,
                "SampleId",
            ]
        )

        actual = labels[idx]

        own_centroid = centroids[
            actual
        ]

        own_std = np.std(
            own_centroid
        )

        sample = X_year[idx]

        if own_std < 1e-12:
            own_similarity = 0.0
        else:
            sample_std = np.std(sample)

            if sample_std < 1e-12:
                own_similarity = 0.0
            else:
                own_similarity = float(
                    np.corrcoef(
                        sample,
                        own_centroid,
                    )[0, 1]
                )

        other_values = []

        for group in groups:

            if group == actual:
                continue

            centroid = centroids[
                group
            ]

            c_std = np.std(
                centroid
            )

            sample_std = np.std(
                sample
            )

            if (
                c_std < 1e-12
                or sample_std < 1e-12
            ):
                similarity = 0.0
            else:
                similarity = float(
                    np.corrcoef(
                        sample,
                        centroid,
                    )[0, 1]
                )

            other_values.append(
                (
                    similarity,
                    group,
                )
            )

        other_values.sort(
            reverse=True
        )

        nearest_similarity, nearest_group = (
            other_values[0]
        )

        sample_rows.append(
            {
                "SampleId": sid,
                "ActualClass": actual,
                "Modality": modality,
                "Year": str(year),
                "OwnGroupSimilarity": own_similarity,
                "NearestOtherGroup": nearest_group,
                "NearestOtherSimilarity": (
                    nearest_similarity
                ),
                "SimilarityGap": (
                    own_similarity
                    - nearest_similarity
                ),
                "OwnGroupWin": (
                    own_similarity
                    >= nearest_similarity
                ),
                "GroupSize": group_sizes[
                    actual
                ],
            }
        )

    sample_df = pd.DataFrame(
        sample_rows
    )

    return (
        pair_df,
        sample_df,
    )


# ==============================================================================
# WITHIN-YEAR STRUCTURE SUMMARY
# ==============================================================================

def structure_summary(
    sample_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:

    if sample_df.empty:
        return pd.DataFrame()

    rows = []

    for (
        modality,
        year,
    ), current in sample_df.groupby(
        [
            "Modality",
            "Year",
        ]
    ):

        pairs = pair_df[
            (
                pair_df["Modality"]
                == modality
            )
            & (
                pair_df["Year"]
                == year
            )
        ]

        rows.append(
            {
                "Modality": modality,
                "Year": year,
                "EligibleSamples": len(
                    current
                ),
                "EligibleGroups": int(
                    current[
                        "ActualClass"
                    ].nunique()
                ),
                "OwnGroupWinRate": float(
                    current[
                        "OwnGroupWin"
                    ].mean()
                ),
                "MeanOwnSimilarity": float(
                    current[
                        "OwnGroupSimilarity"
                    ].mean()
                ),
                "MeanNearestOtherSimilarity": (
                    float(
                        current[
                            "NearestOtherSimilarity"
                        ].mean()
                    )
                ),
                "MeanSimilarityGap": float(
                    current[
                        "SimilarityGap"
                    ].mean()
                ),
                "TopCentroidOverlap": (
                    float(
                        pairs[
                            "CentroidCorrelationSimilarity"
                        ].max()
                    )
                    if not pairs.empty
                    else np.nan
                ),
                "MeanCentroidSimilarity": (
                    float(
                        pairs[
                            "CentroidCorrelationSimilarity"
                        ].mean()
                    )
                    if not pairs.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# GROUP COUNTS WITHIN YEAR
# ==============================================================================

def build_year_group_counts(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    return (
        metadata
        .groupby(
            [
                "Year",
                "ActualClass",
            ]
        )
        .size()
        .reset_index(
            name="Count"
        )
        .sort_values(
            [
                "Year",
                "ActualClass",
            ]
        )
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print("=" * 78)
    print(
        "WITHIN-YEAR GEOGRAPHY DIAGNOSTICS"
    )
    print("=" * 78)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    mapping = load_mapping()

    independent_ids = get_independent_ids(
        mapping
    )

    metadata = build_sample_metadata(
        mapping,
        independent_ids,
    )

    years = sorted(
        metadata["Year"]
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"\nIndependent samples: "
        f"{len(independent_ids)}"
    )

    print(
        f"Years: {', '.join(years)}"
    )

    print("\n")
    print("=" * 78)
    print("WITHIN-YEAR GROUP COUNTS")
    print("=" * 78)

    year_counts = (
        build_year_group_counts(
            metadata
        )
    )

    print(
        year_counts.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Load / process
    # ------------------------------------------------------------------

    processed = load_processed_matrices(
        mapping
    )

    modalities = build_modalities(
        processed
    )

    # ------------------------------------------------------------------
    # Analysis containers
    # ------------------------------------------------------------------

    all_scores = []
    all_variance = []
    all_pca_structure = []
    all_silhouettes = []
    all_pairs = []
    all_samples = []

    # ------------------------------------------------------------------
    # PCA + within-year geography
    # ------------------------------------------------------------------

    for modality, X in modalities.items():

        print("\n")
        print("-" * 78)
        print(
            f"MODALITY: {modality}"
        )
        print("-" * 78)

        for year in years:

            print(
                f"\nYear {year}"
            )

            score_df, variance_df = (
                pca_within_year(
                    X,
                    metadata,
                    modality,
                    year,
                )
            )

            if score_df.empty:
                print(
                    "  Not enough samples."
                )
                continue

            all_scores.append(
                score_df
            )

            all_variance.append(
                variance_df
            )

            # ----------------------------------------------------------
            # PCA geography eta²
            # ----------------------------------------------------------

            pc_structure = (
                calculate_pca_structure(
                    score_df
                )
            )

            all_pca_structure.append(
                pc_structure
            )

            # ----------------------------------------------------------
            # Silhouette
            # ----------------------------------------------------------

            silhouette_df = (
                calculate_silhouette(
                    score_df
                )
            )

            if not silhouette_df.empty:
                all_silhouettes.append(
                    silhouette_df
                )

            # ----------------------------------------------------------
            # Centroid geometry
            # ----------------------------------------------------------

            pair_df, sample_df = (
                centroid_structure(
                    X,
                    metadata,
                    independent_ids,
                    modality,
                    year,
                )
            )

            if not pair_df.empty:
                all_pairs.append(
                    pair_df
                )

            if not sample_df.empty:
                all_samples.append(
                    sample_df
                )

            # ----------------------------------------------------------
            # Print quick result
            # ----------------------------------------------------------

            first_three = variance_df.head(
                3
            )

            print(
                "  PCA first 3 PCs:"
            )

            print(
                first_three[
                    [
                        "PC",
                        "ExplainedVariance",
                        "CumulativeVariance",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

            if not silhouette_df.empty:
                print(
                    "  Geography silhouette:"
                )

                print(
                    silhouette_df.to_string(
                        index=False,
                        float_format=lambda x: f"{x:.4f}",
                    )
                )

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------

    scores_df = (
        pd.concat(
            all_scores,
            ignore_index=True,
        )
        if all_scores
        else pd.DataFrame()
    )

    variance_df = (
        pd.concat(
            all_variance,
            ignore_index=True,
        )
        if all_variance
        else pd.DataFrame()
    )

    pca_structure_df = (
        pd.concat(
            all_pca_structure,
            ignore_index=True,
        )
        if all_pca_structure
        else pd.DataFrame()
    )

    silhouette_df = (
        pd.concat(
            all_silhouettes,
            ignore_index=True,
        )
        if all_silhouettes
        else pd.DataFrame()
    )

    pair_df = (
        pd.concat(
            all_pairs,
            ignore_index=True,
        )
        if all_pairs
        else pd.DataFrame()
    )

    sample_df = (
        pd.concat(
            all_samples,
            ignore_index=True,
        )
        if all_samples
        else pd.DataFrame()
    )

    summary_df = structure_summary(
        sample_df,
        pair_df,
    )

    # ------------------------------------------------------------------
    # PRINT MAIN RESULTS
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "WITHIN-YEAR GEOGRAPHY SUMMARY"
    )
    print("=" * 78)

    if summary_df.empty:
        print(
            "No valid within-year geometry."
        )
    else:
        print(
            summary_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # PCA ETA SUMMARY
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "PCA GEOGRAPHY Eta² WITHIN YEAR"
    )
    print("=" * 78)

    if pca_structure_df.empty:
        print("No PCA structure results.")
    else:

        # Show strongest PCs per modality/year
        top_eta = (
            pca_structure_df
            .sort_values(
                [
                    "Modality",
                    "Year",
                    "GeographyEtaSquared",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            .groupby(
                [
                    "Modality",
                    "Year",
                ],
                as_index=False,
            )
            .head(3)
        )

        print(
            top_eta.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # SILHOUETTE
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "WITHIN-YEAR GEOGRAPHY SILHOUETTE"
    )
    print("=" * 78)

    if silhouette_df.empty:
        print(
            "No valid silhouette results."
        )
    else:
        print(
            silhouette_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # TOP OVERLAPS
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "TOP WITHIN-YEAR GROUP OVERLAPS"
    )
    print("=" * 78)

    if pair_df.empty:
        print(
            "No valid pairwise overlap results."
        )
    else:

        print(
            pair_df.head(40).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # SAMPLES THAT PREFER OTHER GROUPS
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "SAMPLES CLOSER TO ANOTHER GROUP "
        "WITHIN SAME YEAR"
    )
    print("=" * 78)

    if sample_df.empty:
        print(
            "No valid sample geometry."
        )
    else:

        wrong = (
            sample_df[
                ~sample_df[
                    "OwnGroupWin"
                ]
            ]
            .sort_values(
                "SimilarityGap",
                ascending=True,
            )
            .head(40)
        )

        if wrong.empty:
            print(
                "All eligible samples "
                "prefer their own group centroid."
            )
        else:
            print(
                wrong.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    paths = {
        "year_group_counts":
            OUTPUT_DIR
            / "within_year_group_counts.csv",

        "pca_scores":
            OUTPUT_DIR
            / "within_year_pca_scores.csv",

        "pca_variance":
            OUTPUT_DIR
            / "within_year_pca_variance.csv",

        "pca_structure":
            OUTPUT_DIR
            / "within_year_pca_geography_structure.csv",

        "silhouette":
            OUTPUT_DIR
            / "within_year_geography_silhouette.csv",

        "group_overlaps":
            OUTPUT_DIR
            / "within_year_group_overlaps.csv",

        "sample_geometry":
            OUTPUT_DIR
            / "within_year_sample_geometry.csv",

        "summary":
            OUTPUT_DIR
            / "within_year_geography_summary.csv",
    }

    year_counts.to_csv(
        paths["year_group_counts"],
        index=False,
        encoding="utf-8-sig",
    )

    scores_df.to_csv(
        paths["pca_scores"],
        index=False,
        encoding="utf-8-sig",
    )

    variance_df.to_csv(
        paths["pca_variance"],
        index=False,
        encoding="utf-8-sig",
    )

    pca_structure_df.to_csv(
        paths["pca_structure"],
        index=False,
        encoding="utf-8-sig",
    )

    silhouette_df.to_csv(
        paths["silhouette"],
        index=False,
        encoding="utf-8-sig",
    )

    pair_df.to_csv(
        paths["group_overlaps"],
        index=False,
        encoding="utf-8-sig",
    )

    sample_df.to_csv(
        paths["sample_geometry"],
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        paths["summary"],
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


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()