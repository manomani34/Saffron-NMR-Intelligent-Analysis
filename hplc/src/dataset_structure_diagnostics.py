from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse.linalg import spsolve


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXCEL_FILE = (
    ROOT
    / "data"
    / "raw"
    / "hplc.xlsx"
)

MAPPING_FILE = (
    ROOT
    / "data"
    / "mapping"
    / "sample_mapping.csv"
)

OUTPUTS_DIR = (
    ROOT
    / "outputs"
)

GROUP_COUNTS_FILE = (
    OUTPUTS_DIR
    / "dataset_structure_group_counts.csv"
)

YEAR_BALANCE_FILE = (
    OUTPUTS_DIR
    / "dataset_structure_year_balance.csv"
)

SUMMARY_FILE = (
    OUTPUTS_DIR
    / "dataset_structure_summary.csv"
)

GROUP_GEOMETRY_FILE = (
    OUTPUTS_DIR
    / "dataset_structure_group_geometry.csv"
)

SAMPLE_GEOMETRY_FILE = (
    OUTPUTS_DIR
    / "dataset_structure_sample_geometry.csv"
)

NEAREST_OVERLAP_FILE = (
    OUTPUTS_DIR
    / "dataset_structure_nearest_overlaps.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SHEETS = {
    "250 nm": "picrocrocin 250",
    "308 nm": "safranal 308",
    "440 nm": "Crocin 440",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(value) -> str:

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
    )


def detect_column(
    columns,
    candidates,
):

    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in normalized:
            return normalized[key]

    for column in columns:

        text = (
            str(column)
            .strip()
            .lower()
        )

        for candidate in candidates:

            candidate_text = (
                str(candidate)
                .strip()
                .lower()
            )

            if candidate_text in text:
                return column

    return None


def extract_sample_id(value):

    if pd.isna(value):
        return None

    try:

        value_float = float(
            value
        )

        if np.isfinite(
            value_float
        ):

            return int(
                value_float
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


# ============================================================
# ASLS
# ============================================================

def asls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 20,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(y)

    if n < 3:
        return np.zeros_like(y)

    D = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )

    w = np.ones(n)

    for _ in range(n_iter):

        W = sparse.diags(
            w,
            0,
            shape=(n, n),
            format="csc",
        )

        Z = (
            W
            + lam * (D.T @ D)
        )

        baseline = spsolve(
            Z,
            w * y,
        )

        w = np.where(
            y > baseline,
            p,
            1.0 - p,
        )

    return baseline


# ============================================================
# SNV
# ============================================================

def snv(
    y: np.ndarray,
) -> np.ndarray:

    y = np.asarray(
        y,
        dtype=float,
    )

    mean = np.mean(y)
    std = np.std(y)

    if (
        not np.isfinite(std)
        or std <= 1e-12
    ):
        return np.zeros_like(y)

    return (
        y - mean
    ) / std


def preprocess_signal(
    y: np.ndarray,
) -> np.ndarray:

    baseline = asls_baseline(
        y,
        lam=1e5,
        p=0.01,
        n_iter=20,
    )

    corrected = (
        y - baseline
    )

    return snv(
        corrected
    )


# ============================================================
# READ ONE SHEET
# ============================================================

def load_sheet(
    sheet_name: str,
):

    raw = pd.read_excel(
        EXCEL_FILE,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    # --------------------------------------------------------
    # Find header row
    # --------------------------------------------------------

    header_row = None

    for row_idx in range(
        min(10, len(raw))
    ):

        values = [
            normalize_text(v).lower()
            for v
            in raw.iloc[row_idx].tolist()
        ]

        if (
            "number" in values
            and "group" in values
        ):

            header_row = row_idx
            break

    if header_row is None:

        raise ValueError(
            "Header row not found in "
            f"sheet '{sheet_name}'."
        )

    header = raw.iloc[
        header_row
    ].tolist()

    number_col = None
    group_col = None

    for idx, value in enumerate(
        header
    ):

        normalized = (
            normalize_text(value)
            .lower()
        )

        if normalized in {
            "number",
            "sample number",
        }:

            number_col = idx

        if normalized == "group":
            group_col = idx

    if number_col is None:
        number_col = 2

    if group_col is None:
        group_col = 3

    # --------------------------------------------------------
    # Time columns are stored in header names.
    # --------------------------------------------------------

    time_columns = []

    for idx, value in enumerate(
        header
    ):

        try:

            t = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            np.isfinite(t)
            and 0.0 < t <= 30.0
        ):

            time_columns.append(
                (
                    idx,
                    t,
                )
            )

    if len(time_columns) < 100:

        raise ValueError(
            f"Too few time columns in "
            f"{sheet_name}: "
            f"{len(time_columns)}"
        )

    time_columns.sort(
        key=lambda x: x[1]
    )

    time = np.array(
        [
            t
            for _, t
            in time_columns
        ],
        dtype=float,
    )

    column_indices = [
        idx
        for idx, _ in time_columns
    ]

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    rows = []

    for row_idx in range(
        header_row + 1,
        raw.shape[0],
    ):

        sample_id = (
            extract_sample_id(
                raw.iloc[
                    row_idx,
                    number_col,
                ]
            )
        )

        if sample_id is None:
            continue

        group = normalize_text(
            raw.iloc[
                row_idx,
                group_col,
            ]
        )

        y = pd.to_numeric(
            raw.iloc[
                row_idx,
                column_indices,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if len(y) != len(time):
            continue

        finite = np.isfinite(y)

        if finite.sum() < 2:
            continue

        if not finite.all():

            y = np.interp(
                time,
                time[finite],
                y[finite],
            )

        rows.append(
            {
                "SampleId":
                    int(sample_id),

                "WorkbookGroup":
                    group,

                "Time":
                    time.copy(),

                "Signal":
                    y.copy(),
            }
        )

    if not rows:

        raise ValueError(
            f"No usable samples found "
            f"in {sheet_name}."
        )

    return rows


# ============================================================
# AUTHORITATIVE MAPPING
# ============================================================

def load_mapping():

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    sample_col = detect_column(
        mapping.columns,
        [
            "SampleId",
            "Sample ID",
            "Sample_ID",
            "sample",
            "number",
        ],
    )

    group_col = detect_column(
        mapping.columns,
        [
            "Group",
            "GroupId",
            "Group ID",
            "Class",
            "Region",
        ],
    )

    year_col = detect_column(
        mapping.columns,
        [
            "Year",
            "HarvestYear",
            "Harvest Year",
            "year",
        ],
    )

    if (
        sample_col is None
        or group_col is None
    ):

        raise ValueError(
            "Could not detect SampleId "
            "and Group columns in mapping."
        )

    result = pd.DataFrame()

    result["SampleId"] = (
        mapping[sample_col]
        .apply(extract_sample_id)
    )

    result["Group"] = (
        mapping[group_col]
        .astype(str)
        .str.strip()
    )

    if year_col is not None:

        result["Year"] = (
            mapping[year_col]
            .astype(str)
            .str.strip()
        )

    else:

        result["Year"] = ""

    result = result[
        result["SampleId"].notna()
    ].copy()

    result["SampleId"] = (
        result["SampleId"]
        .astype(int)
    )

    result = (
        result
        .drop_duplicates(
            subset=["SampleId"]
        )
        .sort_values("SampleId")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def detect_duplicate_ids(
    all_data,
):

    sample_ids = sorted(
        all_data[
            "SampleId"
        ].unique()
    )

    duplicate_pairs = []

    for i in range(
        len(sample_ids)
    ):

        sid_a = sample_ids[i]

        row_a = all_data[
            all_data["SampleId"]
            == sid_a
        ].sort_values(
            "Wavelength"
        )

        for j in range(
            i + 1,
            len(sample_ids),
        ):

            sid_b = sample_ids[j]

            row_b = all_data[
                all_data["SampleId"]
                == sid_b
            ].sort_values(
                "Wavelength"
            )

            if len(row_a) != len(
                row_b
            ):
                continue

            identical = True

            for wavelength in [
                "250 nm",
                "308 nm",
                "440 nm",
            ]:

                a = (
                    row_a[
                        row_a[
                            "Wavelength"
                        ]
                        == wavelength
                    ]
                    ["Signal"]
                    .iloc[0]
                )

                b = (
                    row_b[
                        row_b[
                            "Wavelength"
                        ]
                        == wavelength
                    ]
                    ["Signal"]
                    .iloc[0]
                )

                if not np.array_equal(
                    a,
                    b,
                ):

                    identical = False
                    break

            if identical:

                duplicate_pairs.append(
                    (
                        sid_a,
                        sid_b,
                    )
                )

    return duplicate_pairs


# ============================================================
# CORRELATION
# ============================================================

def pearson_corr(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if len(a) != len(b):
        return np.nan

    sa = np.std(a)
    sb = np.std(b)

    if (
        sa <= 1e-12
        or sb <= 1e-12
    ):

        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


# ============================================================
# BUILD MODALITY MATRICES
# ============================================================

def build_matrices(
    all_data,
    independent_ids,
):

    matrices = {}
    times = {}

    for wavelength in [
        "250 nm",
        "308 nm",
        "440 nm",
    ]:

        subset = all_data[
            (
                all_data[
                    "Wavelength"
                ]
                == wavelength
            )
            & (
                all_data[
                    "SampleId"
                ].isin(
                    independent_ids
                )
            )
        ].copy()

        subset = (
            subset
            .sort_values("SampleId")
        )

        sample_order = (
            subset[
                "SampleId"
            ]
            .astype(int)
            .tolist()
        )

        signals = []

        for _, row in (
            subset.iterrows()
        ):

            processed = (
                preprocess_signal(
                    row["Signal"]
                )
            )

            signals.append(
                processed
            )

        matrices[wavelength] = (
            sample_order,
            np.vstack(signals),
        )

        times[wavelength] = (
            subset[
                "Time"
            ].iloc[0]
        )

    common_ids = sorted(
        set(
            matrices["250 nm"][0]
        )
        & set(
            matrices["308 nm"][0]
        )
        & set(
            matrices["440 nm"][0]
        )
    )

    combined_blocks = []

    for wavelength in [
        "250 nm",
        "308 nm",
        "440 nm",
    ]:

        ids, matrix = (
            matrices[wavelength]
        )

        index_map = {
            sid: idx
            for idx, sid
            in enumerate(ids)
        }

        indices = [
            index_map[sid]
            for sid in common_ids
        ]

        combined_blocks.append(
            matrix[indices]
        )

    matrices["Combined"] = (
        common_ids,
        np.hstack(
            combined_blocks
        ),
    )

    return (
        matrices,
        times,
        common_ids,
    )


# ============================================================
# SAMPLE-LEVEL GEOMETRY
# ============================================================

def sample_geometry(
    matrix,
    sample_ids,
    groups,
    modality,
):

    id_to_index = {
        sid: idx
        for idx, sid
        in enumerate(
            sample_ids
        )
    }

    group_to_indices = {}

    for idx, sid in enumerate(
        sample_ids
    ):

        group = groups[
            idx
        ]

        group_to_indices.setdefault(
            group,
            [],
        ).append(idx)

    rows = []

    for idx, sid in enumerate(
        sample_ids
    ):

        actual_group = groups[
            idx
        ]

        # ----------------------------------------------------
        # Leave-one-out own centroid
        # ----------------------------------------------------

        own_indices = [
            j
            for j
            in group_to_indices[
                actual_group
            ]
            if j != idx
        ]

        own_similarity = np.nan

        if own_indices:

            own_centroid = np.mean(
                matrix[
                    own_indices
                ],
                axis=0,
            )

            own_similarity = (
                pearson_corr(
                    matrix[idx],
                    own_centroid,
                )
            )

        # ----------------------------------------------------
        # All other group centroids
        # ----------------------------------------------------

        other_scores = []

        for group, indices in (
            group_to_indices.items()
        ):

            if group == actual_group:
                continue

            centroid = np.mean(
                matrix[indices],
                axis=0,
            )

            similarity = (
                pearson_corr(
                    matrix[idx],
                    centroid,
                )
            )

            other_scores.append(
                (
                    group,
                    similarity,
                )
            )

        if other_scores:

            other_scores = [
                x
                for x
                in other_scores
                if np.isfinite(
                    x[1]
                )
            ]

        if other_scores:

            other_scores.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            nearest_other_group = (
                other_scores[0][0]
            )

            nearest_other_similarity = (
                other_scores[0][1]
            )

        else:

            nearest_other_group = ""
            nearest_other_similarity = np.nan

        gap = np.nan

        if (
            np.isfinite(
                own_similarity
            )
            and np.isfinite(
                nearest_other_similarity
            )
        ):

            gap = (
                own_similarity
                - nearest_other_similarity
            )

        own_wins = (
            bool(
                own_similarity
                >= nearest_other_similarity
            )
            if (
                np.isfinite(
                    own_similarity
                )
                and np.isfinite(
                    nearest_other_similarity
                )
            )
            else False
        )

        rows.append(
            {
                "SampleId":
                    int(sid),

                "ActualClass":
                    actual_group,

                "Modality":
                    modality,

                "OwnGroupSimilarity":
                    own_similarity,

                "NearestOtherGroup":
                    nearest_other_group,

                "NearestOtherSimilarity":
                    nearest_other_similarity,

                "SimilarityGap":
                    gap,

                "OwnGroupWins":
                    own_wins,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GROUP-LEVEL GEOMETRY
# ============================================================

def group_geometry(
    matrix,
    sample_ids,
    groups,
    modality,
):

    group_names = sorted(
        np.unique(groups)
    )

    group_to_indices = {}

    for idx, group in enumerate(
        groups
    ):

        group_to_indices.setdefault(
            group,
            [],
        ).append(idx)

    rows = []

    # --------------------------------------------------------
    # Overall pairwise distances
    # --------------------------------------------------------

    within_distances = []
    between_distances = []

    for i in range(
        len(sample_ids)
    ):

        for j in range(
            i + 1,
            len(sample_ids),
        ):

            distance = float(
                np.linalg.norm(
                    matrix[i]
                    - matrix[j]
                )
            )

            if (
                groups[i]
                == groups[j]
            ):

                within_distances.append(
                    distance
                )

            else:

                between_distances.append(
                    distance
                )

    overall_within = (
        float(
            np.mean(
                within_distances
            )
        )
        if within_distances
        else np.nan
    )

    overall_between = (
        float(
            np.mean(
                between_distances
            )
        )
        if between_distances
        else np.nan
    )

    ratio = np.nan

    if (
        np.isfinite(
            overall_within
        )
        and np.isfinite(
            overall_between
        )
        and overall_between > 0
    ):

        ratio = (
            overall_within
            / overall_between
        )

    # --------------------------------------------------------
    # Group centroids
    # --------------------------------------------------------

    centroids = {}

    for group in group_names:

        indices = (
            group_to_indices[group]
        )

        centroids[group] = np.mean(
            matrix[indices],
            axis=0,
        )

    # --------------------------------------------------------
    # Group geometry
    # --------------------------------------------------------

    for group in group_names:

        indices = (
            group_to_indices[group]
        )

        centroid = centroids[
            group
        ]

        # within-group sample distances
        group_distances = []

        for i in indices:

            for j in indices:

                if j <= i:
                    continue

                distance = float(
                    np.linalg.norm(
                        matrix[i]
                        - matrix[j]
                    )
                )

                group_distances.append(
                    distance
                )

        within_mean = (
            float(
                np.mean(
                    group_distances
                )
            )
            if group_distances
            else np.nan
        )

        # sample-to-centroid radius
        radii = []

        for i in indices:

            radius = float(
                np.linalg.norm(
                    matrix[i]
                    - centroid
                )
            )

            radii.append(
                radius
            )

        radius_mean = (
            float(
                np.mean(radii)
            )
            if radii
            else np.nan
        )

        # nearest foreign centroid
        foreign = []

        for other_group in (
            group_names
        ):

            if other_group == group:
                continue

            similarity = (
                pearson_corr(
                    centroid,
                    centroids[
                        other_group
                    ],
                )
            )

            foreign.append(
                (
                    other_group,
                    similarity,
                )
            )

        foreign.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        if foreign:

            nearest_group = (
                foreign[0][0]
            )

            nearest_similarity = (
                foreign[0][1]
            )

        else:

            nearest_group = ""
            nearest_similarity = np.nan

        rows.append(
            {
                "Modality":
                    modality,

                "Group":
                    group,

                "SampleCount":
                    len(indices),

                "WithinGroupMeanDistance":
                    within_mean,

                "CentroidRadiusMean":
                    radius_mean,

                "NearestOtherGroup":
                    nearest_group,

                "NearestOtherCentroidSimilarity":
                    nearest_similarity,

                "OverallWithinDistance":
                    overall_within,

                "OverallBetweenDistance":
                    overall_between,

                "WithinBetweenDistanceRatio":
                    ratio,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# OVERLAPPING GROUP PAIRS
# ============================================================

def group_pair_overlaps(
    matrix,
    sample_ids,
    groups,
    modality,
):

    group_names = sorted(
        np.unique(groups)
    )

    group_to_indices = {}

    for idx, group in enumerate(
        groups
    ):

        group_to_indices.setdefault(
            group,
            [],
        ).append(idx)

    centroids = {
        group: np.mean(
            matrix[
                group_to_indices[group]
            ],
            axis=0,
        )
        for group in group_names
    }

    rows = []

    for i in range(
        len(group_names)
    ):

        for j in range(
            i + 1,
            len(group_names),
        ):

            g1 = group_names[i]
            g2 = group_names[j]

            similarity = (
                pearson_corr(
                    centroids[g1],
                    centroids[g2],
                )
            )

            distance = float(
                np.linalg.norm(
                    centroids[g1]
                    - centroids[g2]
                )
            )

            rows.append(
                {
                    "Modality":
                        modality,

                    "GroupA":
                        g1,

                    "GroupB":
                        g2,

                    "CentroidSimilarity":
                        similarity,

                    "CentroidDistance":
                        distance,
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "Modality",
            "CentroidSimilarity",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "HPLC DATASET STRUCTURE DIAGNOSTICS"
    )
    print(
        "=" * 78
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found:\n"
            f"{EXCEL_FILE}"
        )

    if not MAPPING_FILE.exists():

        raise FileNotFoundError(
            f"Mapping file not found:\n"
            f"{MAPPING_FILE}"
        )

    # ========================================================
    # MAPPING
    # ========================================================

    mapping = load_mapping()

    print()
    print(
        f"Mapping rows: "
        f"{len(mapping)}"
    )

    # ========================================================
    # LOAD ALL THREE SHEETS
    # ========================================================

    raw_rows = []

    for modality, sheet_name in (
        SHEETS.items()
    ):

        print()
        print(
            f"Loading {modality}..."
        )

        rows = load_sheet(
            sheet_name
        )

        for row in rows:

            raw_rows.append(
                {
                    "SampleId":
                        row["SampleId"],

                    "Wavelength":
                        modality,

                    "WorkbookGroup":
                        row["WorkbookGroup"],

                    "Time":
                        row["Time"],

                    "Signal":
                        row["Signal"],
                }
            )

        print(
            f"  Samples: {len(rows)}"
        )

        print(
            f"  Points : "
            f"{len(rows[0]['Time'])}"
        )

        print(
            f"  Time   : "
            f"{rows[0]['Time'].min():.6f}"
            f" -> "
            f"{rows[0]['Time'].max():.6f}"
        )

    all_data = pd.DataFrame(
        raw_rows
    )

    # ========================================================
    # MERGE AUTHORITATIVE MAPPING
    # ========================================================

    all_data = all_data.merge(
        mapping[
            [
                "SampleId",
                "Group",
                "Year",
            ]
        ],
        on="SampleId",
        how="left",
    )

    # ========================================================
    # DUPLICATES
    # ========================================================

    print()
    print(
        "Detecting exact duplicate measurements..."
    )

    duplicate_pairs = (
        detect_duplicate_ids(
            all_data
        )
    )

    print(
        f"Duplicate pairs found: "
        f"{len(duplicate_pairs)}"
    )

    for pair in duplicate_pairs:

        print(
            f"  {pair[0]} == {pair[1]}"
        )

    duplicate_remove_ids = {
        pair[1]
        for pair
        in duplicate_pairs
    }

    independent_ids = sorted(
        set(
            mapping[
                "SampleId"
            ].astype(int)
        )
        - duplicate_remove_ids
    )

    print()
    print(
        f"Raw sample IDs       : "
        f"{len(mapping)}"
    )

    print(
        f"Independent sample IDs: "
        f"{len(independent_ids)}"
    )

    # ========================================================
    # GROUP COUNTS
    # ========================================================

    independent_mapping = mapping[
        mapping["SampleId"].isin(
            independent_ids
        )
    ].copy()

    group_counts = (
        independent_mapping[
            "Group"
        ]
        .value_counts()
        .rename_axis("Group")
        .reset_index(
            name="IndependentSamples"
        )
        .sort_values(
            "Group"
        )
    )

    group_counts[
        "Percent"
    ] = (
        group_counts[
            "IndependentSamples"
        ]
        / len(independent_ids)
        * 100.0
    )

    group_counts.to_csv(
        GROUP_COUNTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 78
    )
    print(
        "GROUP SIZE"
    )
    print(
        "=" * 78
    )

    print(
        group_counts.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    # ========================================================
    # YEAR BALANCE
    # ========================================================

    year_balance = (
        independent_mapping[
            [
                "Group",
                "Year",
            ]
        ]
        .value_counts()
        .rename(
            "Count"
        )
        .reset_index()
        .sort_values(
            [
                "Group",
                "Year",
            ]
        )
    )

    year_balance.to_csv(
        YEAR_BALANCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "=" * 78
    )
    print(
        "GROUP × YEAR"
    )
    print(
        "=" * 78
    )

    print(
        year_balance.to_string(
            index=False
        )
    )

    # ========================================================
    # MATRICES
    # ========================================================

    (
        matrices,
        times,
        common_ids,
    ) = build_matrices(
        all_data,
        independent_ids,
    )

    # authoritative groups
    group_lookup = {
        int(row["SampleId"]):
            str(row["Group"])
        for _, row
        in independent_mapping.iterrows()
    }

    # ========================================================
    # GEOMETRY ANALYSIS
    # ========================================================

    all_summary = []
    all_group_geometry = []
    all_sample_geometry = []
    all_overlaps = []

    modalities = [
        "250 nm",
        "308 nm",
        "440 nm",
        "Combined",
    ]

    for modality in modalities:

        sample_ids, matrix = (
            matrices[modality]
        )

        groups = np.array(
            [
                group_lookup[
                    int(sid)
                ]
                for sid in sample_ids
            ]
        )

        # ----------------------------------------------------
        # Sample geometry
        # ----------------------------------------------------

        sample_df = (
            sample_geometry(
                matrix,
                sample_ids,
                groups,
                modality,
            )
        )

        sample_df[
            "GroupSize"
        ] = sample_df[
            "ActualClass"
        ].map(
            group_counts.set_index(
                "Group"
            )[
                "IndependentSamples"
            ]
        )

        # ----------------------------------------------------
        # Group geometry
        # ----------------------------------------------------

        group_df = (
            group_geometry(
                matrix,
                sample_ids,
                groups,
                modality,
            )
        )

        # ----------------------------------------------------
        # Pair overlap
        # ----------------------------------------------------

        overlap_df = (
            group_pair_overlaps(
                matrix,
                sample_ids,
                groups,
                modality,
            )
        )

        # ----------------------------------------------------
        # Overall summary
        # ----------------------------------------------------

        own_wins = (
            sample_df[
                "OwnGroupWins"
            ]
            .mean()
        )

        mean_own_similarity = (
            sample_df[
                "OwnGroupSimilarity"
            ]
            .mean()
        )

        mean_best_other = (
            sample_df[
                "NearestOtherSimilarity"
            ]
            .mean()
        )

        mean_gap = (
            sample_df[
                "SimilarityGap"
            ]
            .mean()
        )

        top_overlap = (
            overlap_df.iloc[0]
            if not overlap_df.empty
            else None
        )

        overall_within = (
            group_df[
                "OverallWithinDistance"
            ].iloc[0]
        )

        overall_between = (
            group_df[
                "OverallBetweenDistance"
            ].iloc[0]
        )

        ratio = (
            group_df[
                "WithinBetweenDistanceRatio"
            ].iloc[0]
        )

        all_summary.append(
            {
                "Modality":
                    modality,

                "Samples":
                    len(sample_ids),

                "Groups":
                    len(
                        np.unique(groups)
                    ),

                "OwnCentroidWinRate":
                    own_wins,

                "MeanOwnCentroidSimilarity":
                    mean_own_similarity,

                "MeanNearestOtherSimilarity":
                    mean_best_other,

                "MeanSimilarityGap":
                    mean_gap,

                "MeanWithinGroupDistance":
                    overall_within,

                "MeanBetweenGroupDistance":
                    overall_between,

                "WithinBetweenDistanceRatio":
                    ratio,

                "MostOverlappingGroupA":
                    (
                        top_overlap["GroupA"]
                        if top_overlap
                        is not None
                        else ""
                    ),

                "MostOverlappingGroupB":
                    (
                        top_overlap["GroupB"]
                        if top_overlap
                        is not None
                        else ""
                    ),

                "MostOverlappingCentroidSimilarity":
                    (
                        top_overlap[
                            "CentroidSimilarity"
                        ]
                        if top_overlap
                        is not None
                        else np.nan
                    ),
            }
        )

        all_group_geometry.append(
            group_df
        )

        all_sample_geometry.append(
            sample_df
        )

        all_overlaps.append(
            overlap_df
        )

    # ========================================================
    # SAVE ALL
    # ========================================================

    summary_df = pd.DataFrame(
        all_summary
    )

    group_geometry_df = (
        pd.concat(
            all_group_geometry,
            ignore_index=True,
        )
    )

    sample_geometry_df = (
        pd.concat(
            all_sample_geometry,
            ignore_index=True,
        )
    )

    overlap_df = (
        pd.concat(
            all_overlaps,
            ignore_index=True,
        )
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    group_geometry_df.to_csv(
        GROUP_GEOMETRY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    sample_geometry_df.to_csv(
        SAMPLE_GEOMETRY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    overlap_df.to_csv(
        NEAREST_OVERLAP_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "STRUCTURE SUMMARY"
    )
    print(
        "=" * 78
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # TOP OVERLAPS
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "TOP GROUP OVERLAPS"
    )
    print(
        "=" * 78
    )

    print(
        overlap_df
        .groupby("Modality")
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # WORST SAMPLE GEOMETRY
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "SAMPLES CLOSER TO ANOTHER GROUP"
    )
    print(
        "=" * 78
    )

    problematic = (
        sample_geometry_df[
            ~sample_geometry_df[
                "OwnGroupWins"
            ]
        ]
        .sort_values(
            "SimilarityGap"
        )
    )

    print(
        problematic.head(30).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "OUTPUTS"
    )
    print(
        "=" * 78
    )

    print(
        f"Group counts : "
        f"{GROUP_COUNTS_FILE}"
    )

    print(
        f"Year balance : "
        f"{YEAR_BALANCE_FILE}"
    )

    print(
        f"Summary      : "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Group geometry: "
        f"{GROUP_GEOMETRY_FILE}"
    )

    print(
        f"Sample geometry: "
        f"{SAMPLE_GEOMETRY_FILE}"
    )

    print(
        f"Overlaps     : "
        f"{NEAREST_OVERLAP_FILE}"
    )

    print()


if __name__ == "__main__":
    main()