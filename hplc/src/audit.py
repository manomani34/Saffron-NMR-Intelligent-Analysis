from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_loader import HPLCData


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _safe_stat(values: np.ndarray, fn, default=np.nan):
    finite = _finite(values)
    if finite.size == 0:
        return default
    try:
        return float(fn(finite))
    except Exception:
        return default


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def wavelength_audit_table(data: HPLCData) -> pd.DataFrame:
    rows = []
    independent_ids = set(
        data.mapping.loc[
            data.mapping["Independent"], "SampleId"
        ].astype(int)
    )

    for code in ["440", "250", "308"]:
        item = data.wavelengths[code]
        values = np.asarray(item.values, dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "Wavelength": f"{code} nm",
                "Rows": int(values.shape[0]),
                "Independent": int(
                    sum(int(s) in independent_ids for s in item.sample_ids)
                ),
                "TimePoints": int(values.shape[1]),
                "MissingPoints": int(np.isnan(values).sum()),
                "NegativePoints": int(np.sum(values < 0)),
                "Min": float(np.min(finite)) if finite.size else np.nan,
                "Median": float(np.median(finite)) if finite.size else np.nan,
                "Mean": float(np.mean(finite)) if finite.size else np.nan,
                "Std": float(np.std(finite)) if finite.size else np.nan,
                "Max": float(np.max(finite)) if finite.size else np.nan,
            }
        )

    return pd.DataFrame(rows)


def duplicate_audit_table(data: HPLCData) -> pd.DataFrame:
    mapping = data.mapping.copy()
    dup_groups = (
        mapping.groupby("HPLCNumber", dropna=False)["SampleId"]
        .agg(list)
        .reset_index()
    )
    dup_groups = dup_groups[dup_groups["SampleId"].map(len) > 1]

    rows: List[dict] = []
    position = {
        code: {int(sid): idx for idx, sid in enumerate(item.sample_ids.tolist())}
        for code, item in data.wavelengths.items()
    }

    for _, group in dup_groups.iterrows():
        sample_ids = [int(x) for x in group["SampleId"]]
        pair_diffs: Dict[str, float] = {}
        exact_flags: List[bool] = []

        for code, item in data.wavelengths.items():
            if len(sample_ids) != 2:
                continue
            a = item.values[position[code][sample_ids[0]]]
            b = item.values[position[code][sample_ids[1]]]
            diff = np.abs(a - b)
            finite = diff[np.isfinite(diff)]
            max_diff = float(np.max(finite)) if finite.size else np.nan
            pair_diffs[f"MaxAbsDiff_{code}"] = max_diff
            exact_flags.append(bool(np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True)))

        rows.append(
            {
                "HPLCNumber": str(group["HPLCNumber"]),
                "SamplePair": "/".join(map(str, sample_ids)),
                **pair_diffs,
                "ExactAcrossAllWavelengths": all(exact_flags),
            }
        )

    return pd.DataFrame(rows)


def independent_sample_summary(data: HPLCData, code: str) -> pd.DataFrame:
    item = data.wavelengths[code]
    mapping = data.mapping[data.mapping["Independent"]].copy()
    position = {
        int(sid): idx for idx, sid in enumerate(item.sample_ids.tolist())
    }

    rows = []
    for _, meta in mapping.iterrows():
        sid = int(meta["SampleId"])
        row = np.asarray(item.values[position[sid]], dtype=float)
        finite_mask = np.isfinite(row)
        finite = row[finite_mask]
        valid_time = item.time[finite_mask]

        if finite.size:
            peak_idx = int(np.nanargmax(row))
            peak_time = float(item.time[peak_idx])
            peak_height = float(row[peak_idx])
            min_value = float(np.min(finite))
            max_value = float(np.max(finite))
            median_value = float(np.median(finite))
            mean_value = float(np.mean(finite))
            std_value = float(np.std(finite))
            auc = _trapz(finite, valid_time)
        else:
            peak_time = peak_height = min_value = max_value = np.nan
            median_value = mean_value = std_value = auc = np.nan

        rows.append(
            {
                "SampleId": sid,
                "HPLCNumber": str(meta["HPLCNumber"]),
                "Group": str(meta["Group"]),
                "Region": str(meta["Region"]),
                "HarvestYear": int(meta["HarvestYear"]),
                "Min": min_value,
                "Median": median_value,
                "Mean": mean_value,
                "Std": std_value,
                "Max": max_value,
                "PeakTimeMin": peak_time,
                "AUC": auc,
                "NegativePoints": int(np.sum(row < 0)),
                "MissingPoints": int(np.isnan(row).sum()),
            }
        )

    return pd.DataFrame(rows)


def robust_outlier_flags(summary_by_wavelength: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []

    for code, df in summary_by_wavelength.items():
        work = df.copy()
        work["logMax"] = np.log1p(
            np.clip(work["Max"], a_min=0, a_max=None)
        )
        work["logAUCAbs"] = np.log1p(
            np.abs(work["AUC"])
        )

        def modified_z(series: pd.Series) -> pd.Series:
            values = pd.to_numeric(series, errors="coerce")
            med = values.median()
            mad = (
                np.median(
                    np.abs(values.dropna() - med)
                )
                if values.notna().any()
                else 0.0
            )
            if not np.isfinite(mad) or mad == 0:
                return pd.Series(
                    np.zeros(len(values)),
                    index=values.index,
                )
            return 0.6745 * (values - med) / mad

        # The raw workbook shows a large signal-scale discontinuity between
        # harvest years. Screening is therefore done within year so that the
        # year-dependent scale does not automatically label one whole year
        # as unusual. This is only a statistical screening flag.
        flagged_parts = []
        for _, year_df in work.groupby("HarvestYear", dropna=False):
            year_df = year_df.copy()
            year_df["PeakHeightRobustZ"] = modified_z(
                year_df["logMax"]
            )
            year_df["AUCRobustZ"] = modified_z(
                year_df["logAUCAbs"]
            )
            year_df["Flag"] = (
                year_df["PeakHeightRobustZ"].abs().ge(3.5)
                | year_df["AUCRobustZ"].abs().ge(3.5)
            )
            flagged_parts.append(year_df)

        work = pd.concat(
            flagged_parts,
            ignore_index=True,
        )
        work["Wavelength"] = f"{code} nm"
        frames.append(
            work[
                [
                    "SampleId",
                    "HPLCNumber",
                    "Group",
                    "Region",
                    "HarvestYear",
                    "Wavelength",
                    "Max",
                    "AUC",
                    "PeakHeightRobustZ",
                    "AUCRobustZ",
                    "Flag",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def year_scale_summary(summary_by_wavelength: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for code, df in summary_by_wavelength.items():
        grouped = df.groupby("HarvestYear")
        max_by_year = grouped["Max"].median()
        auc_by_year = grouped["AUC"].median()

        median_max_1394 = float(max_by_year.get(1394, np.nan))
        median_max_1404 = float(max_by_year.get(1404, np.nan))
        median_auc_1394 = float(auc_by_year.get(1394, np.nan))
        median_auc_1404 = float(auc_by_year.get(1404, np.nan))

        rows.append(
            {
                "Wavelength": f"{code} nm",
                "MedianMax_1394": median_max_1394,
                "MedianMax_1404": median_max_1404,
                "MedianMaxRatio_1394_to_1404": (
                    median_max_1394 / median_max_1404
                    if np.isfinite(median_max_1394)
                    and np.isfinite(median_max_1404)
                    and median_max_1404 != 0
                    else np.nan
                ),
                "MedianAUC_1394": median_auc_1394,
                "MedianAUC_1404": median_auc_1404,
                "MedianAUCRatio_1394_to_1404": (
                    median_auc_1394 / median_auc_1404
                    if np.isfinite(median_auc_1394)
                    and np.isfinite(median_auc_1404)
                    and median_auc_1404 != 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def full_audit(data: HPLCData) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    wavelength = wavelength_audit_table(data)
    duplicates = duplicate_audit_table(data)
    by_wave = {
        code: independent_sample_summary(data, code)
        for code in ["440", "250", "308"]
    }
    outliers = robust_outlier_flags(by_wave)
    year_scale = year_scale_summary(by_wave)
    return wavelength, duplicates, by_wave, outliers, year_scale
