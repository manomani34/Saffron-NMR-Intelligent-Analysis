from __future__ import annotations

import pandas as pd


def fmt(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def fmt_pm(mean, std):
    try:
        mean = float(mean)
    except Exception:
        return "—"

    try:
        std = float(std)
    except Exception:
        std = 0.0

    return f"{mean:.4f} ± {std:.4f}"


def mapping_summary(mapping: pd.DataFrame) -> dict:
    independent = mapping[mapping["Independent"]]
    return {
        "rows": int(len(mapping)),
        "independent": int(len(independent)),
        "groups": int(mapping["Group"].nunique()),
        "year_1394": int((independent["HarvestYear"] == 1394).sum()),
        "year_1404": int((independent["HarvestYear"] == 1404).sum()),
    }
