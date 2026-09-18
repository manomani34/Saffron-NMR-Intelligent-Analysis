from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_TIME_POINTS,
    HPLC_RAW_PATH,
    MAPPING_PATH,
    SHEETS,
    TIME_MAX,
    TIME_MIN,
)


@dataclass
class HPLCWavelength:
    code: str
    sheet_name: str
    time: np.ndarray
    sample_ids: np.ndarray
    groups: List[str]
    values: np.ndarray


@dataclass
class HPLCData:
    wavelengths: Dict[str, HPLCWavelength]
    mapping: pd.DataFrame
    group_mismatches: pd.DataFrame


def _find_header_row(raw: pd.DataFrame) -> int:
    for r in range(min(8, len(raw))):
        row = raw.iloc[r].astype(str).str.strip().str.lower()
        if row.isin({"group", "group "}).any():
            return r
    raise ValueError("Could not locate the HPLC header row.")


def _parse_sheet(path: Path, code: str, sheet_name: str) -> HPLCWavelength:
    raw = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    header_row = _find_header_row(raw)
    header = raw.iloc[header_row]

    time_cols = []
    time_values = []

    for col in range(4, raw.shape[1]):
        value = header.iloc[col]
        try:
            t = float(value)
        except (TypeError, ValueError):
            continue

        if TIME_MIN - 1e-8 <= t <= TIME_MAX + 1e-8:
            time_cols.append(col)
            time_values.append(t)

    if len(time_cols) != EXPECTED_TIME_POINTS:
        raise ValueError(
            f"{sheet_name}: expected {EXPECTED_TIME_POINTS} time points "
            f"within 0-30 min, found {len(time_cols)}."
        )

    sample_rows = raw.iloc[header_row + 1 :].copy()

    sample_ids = pd.to_numeric(
        sample_rows.iloc[:, 2],
        errors="coerce",
    )
    groups = (
        sample_rows.iloc[:, 3]
        .astype(str)
        .str.strip()
        .tolist()
    )

    valid = sample_ids.notna()
    sample_ids = sample_ids.loc[valid].astype(int).to_numpy()
    groups = [g for g, ok in zip(groups, valid.tolist()) if ok]

    values = (
        sample_rows.loc[valid, time_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    return HPLCWavelength(
        code=code,
        sheet_name=sheet_name,
        time=np.asarray(time_values, dtype=float),
        sample_ids=sample_ids,
        groups=groups,
        values=values,
    )


def _load_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(
        path,
        dtype={
            "Group": str,
            "Region": str,
            "SampleName": str,
            "HPLCNumber": str,
        },
    )

    mapping["SampleId"] = pd.to_numeric(
        mapping["SampleId"],
        errors="raise",
    ).astype(int)

    mapping["HarvestYear"] = pd.to_numeric(
        mapping["HarvestYear"],
        errors="raise",
    ).astype(int)

    mapping["Independent"] = (
        mapping["Independent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    mapping = mapping.sort_values("SampleId").reset_index(drop=True)

    if len(mapping) != 44:
        raise ValueError(
            f"Expected 44 mapping rows, found {len(mapping)}."
        )

    return mapping


def _validate_against_mapping(
    wavelengths: Dict[str, HPLCWavelength],
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    expected_ids = mapping["SampleId"].tolist()
    mismatch_rows = []

    for code, item in wavelengths.items():
        ids = item.sample_ids.tolist()

        if ids != expected_ids:
            raise ValueError(
                f"Wavelength {code}: sample order does not match the "
                "authoritative mapping."
            )

        mapping_groups = dict(
            zip(
                mapping["SampleId"].astype(int),
                mapping["Group"].astype(str).str.strip(),
            )
        )

        for sample_id, workbook_group in zip(
            item.sample_ids,
            item.groups,
        ):
            workbook_group = str(workbook_group).strip()
            mapping_group = mapping_groups[int(sample_id)]

            if workbook_group != mapping_group:
                mismatch_rows.append(
                    {
                        "Wavelength": code,
                        "SampleId": int(sample_id),
                        "WorkbookGroup": workbook_group,
                        "MappingGroup": mapping_group,
                    }
                )

        if not np.allclose(
            item.time,
            wavelengths["440"].time,
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError(
                f"Wavelength {code}: time axis differs from 440 nm."
            )

    return pd.DataFrame(mismatch_rows)


def load_hplc_data(
    path: Path = HPLC_RAW_PATH,
    mapping_path: Path = MAPPING_PATH,
) -> HPLCData:
    mapping = _load_mapping(mapping_path)

    wavelengths = {
        code: _parse_sheet(path, code, sheet_name)
        for code, sheet_name in SHEETS.items()
    }

    group_mismatches = _validate_against_mapping(
        wavelengths,
        mapping,
    )

    return HPLCData(
        wavelengths=wavelengths,
        mapping=mapping,
        group_mismatches=group_mismatches,
    )
