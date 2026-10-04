from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HPLC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HPLC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hplc.src.audit import full_audit
from hplc.src.config import (
    ASLS_ITERATIONS,
    ASLS_LAMBDA,
    ASLS_P,
    CV_REPEATS,
    CV_N_SPLITS,
    RANDOM_STATE,
    HPLC_RAW_PATH,
    MAPPING_PATH,
    OUTPUTS_DIR,
)
from hplc.src.data_loader import load_hplc_data
from hplc.src.modeling import (
    evaluate_origin_pipeline,
    fit_descriptive_models,
    summarize_origin_results,
    summarize_repeat_results,
)


def write_csv(df: pd.DataFrame, filename: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        OUTPUTS_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def write_matrix_bundle(data) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    independent = data.mapping[data.mapping["Independent"]].copy()
    positions = {
        code: {
            int(sample_id): idx
            for idx, sample_id in enumerate(item.sample_ids.tolist())
        }
        for code, item in data.wavelengths.items()
    }
    ids = independent["SampleId"].astype(int).tolist()

    matrix_payload = {}
    for code in ["440", "250", "308"]:
        item = data.wavelengths[code]
        idx = [positions[code][sid] for sid in ids]
        matrix_payload[f"X_{code}"] = item.values[idx]

    np.savez_compressed(
        OUTPUTS_DIR / "independent_raw_matrices.npz",
        SampleId=np.asarray(ids, dtype=int),
        Time440=data.wavelengths["440"].time,
        Time250=data.wavelengths["250"].time,
        Time308=data.wavelengths["308"].time,
        **matrix_payload,
    )


def run_pipeline() -> None:
    started = datetime.now().isoformat(timespec="seconds")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HPLC SAFFRON ANALYSIS PIPELINE")
    print("=" * 72)
    print(f"Raw workbook : {HPLC_RAW_PATH}")
    print(f"Mapping      : {MAPPING_PATH}")
    print()

    print("[1/5] Loading and validating raw HPLC data...")
    data = load_hplc_data(HPLC_RAW_PATH, MAPPING_PATH)

    print("[2/5] Running data audit...")
    wavelength_audit, duplicate_audit, summaries, outliers, year_scale = full_audit(data)

    write_csv(wavelength_audit, "data_audit_wavelengths.csv")
    write_csv(duplicate_audit, "duplicate_audit.csv")
    write_csv(data.group_mismatches, "metadata_mismatches.csv")
    write_csv(year_scale, "year_scale_summary.csv")
    for code, df in summaries.items():
        write_csv(df, f"sample_signal_audit_{code}.csv")
    write_csv(outliers, "signal_outlier_screening.csv")
    write_matrix_bundle(data)

    print("    Audit outputs written.")

    independent = data.mapping[data.mapping["Independent"]].copy().reset_index(drop=True)
    sample_ids = independent["SampleId"].astype(int).to_numpy()
    y = independent["Group"].astype(str).to_numpy()

    positions = {
        code: {
            int(sample_id): idx
            for idx, sample_id in enumerate(item.sample_ids.tolist())
        }
        for code, item in data.wavelengths.items()
    }

    X_by_modality: dict[str, np.ndarray] = {}
    for code, label in [("440", "440 nm"), ("250", "250 nm"), ("308", "308 nm")]:
        item = data.wavelengths[code]
        idx = [positions[code][sid] for sid in sample_ids]
        X_by_modality[label] = np.asarray(item.values[idx], dtype=float)

    # Provenance-aware CV units. Samples 39 and 40 are a deterministic
    # scaled-copy pair in 308 nm (40 = 39 * 0.53). They must stay in the
    # same train/test partition for 308 nm and Combined. The same splits are
    # reused across modalities so modality comparisons remain paired.
    provenance_groups = sample_ids.astype(str).astype(object)
    linked = {39: "prov_308_39_40", 40: "prov_308_39_40"}
    provenance_groups = np.asarray(
        [linked.get(int(sid), str(int(sid))) for sid in sample_ids],
        dtype=object,
    )

    print(
        f"[3/5] Running provenance-aware origin evaluation "
        f"(StratifiedGroupKFold {CV_N_SPLITS}-Fold x {CV_REPEATS})..."
    )
    split_results, cv_predictions = evaluate_origin_pipeline(
        X_by_modality=X_by_modality,
        y=y,
        sample_ids=sample_ids,
        repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
        asls_lambda=ASLS_LAMBDA,
        asls_p=ASLS_P,
        asls_iterations=ASLS_ITERATIONS,
        provenance_groups=provenance_groups,
    )

    fold_summary = summarize_origin_results(split_results)
    summary = summarize_repeat_results(cv_predictions)

    # Final reporting uses repeat-pooled OOF metrics. Fold-level metrics are
    # retained separately for diagnostics. This avoids treating the fold that
    # lacks G9 as if it were a complete 11-class test fold.
    write_csv(split_results, "robust_evaluation_splits.csv")
    write_csv(fold_summary, "robust_evaluation_fold_summary.csv")
    write_csv(summary, "robust_evaluation_summary.csv")
    write_csv(cv_predictions, "robust_evaluation_predictions.csv")

    write_csv(split_results, "origin_cv_splits.csv")
    write_csv(summary, "origin_cv_summary.csv")
    write_csv(cv_predictions, "origin_cv_predictions.csv")

    sample_cv_summary = (
        cv_predictions.groupby(
            [
                "Preprocessing",
                "PreprocessingCode",
                "Modality",
                "SampleId",
                "ActualGroup",
            ],
            as_index=False,
        )
        .agg(
            MostFrequentPredictedGroup=(
                "PredictedGroup",
                lambda s: s.mode().iloc[0] if not s.mode().empty else "",
            ),
            CorrectRate=("Correct", "mean"),
            MeanDecisionMargin=("DecisionMargin", "mean"),
            PredictionCount=("Correct", "size"),
        )
    )
    write_csv(sample_cv_summary, "robust_sample_cv_summary.csv")
    write_csv(sample_cv_summary, "sample_cv_summary.csv")

    print("[4/5] Building descriptive full-data predictions...")
    descriptive = fit_descriptive_models(
        X_by_modality=X_by_modality,
        y=y,
        sample_ids=sample_ids,
        preprocessing_method="asls_snv",
        asls_lambda=ASLS_LAMBDA,
        asls_p=ASLS_P,
        asls_iterations=ASLS_ITERATIONS,
    )
    write_csv(descriptive, "descriptive_predictions.csv")

    print("[5/5] Writing pipeline manifest...")
    manifest = {
        "pipeline": "HPLC saffron origin analysis",
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "raw_workbook": str(HPLC_RAW_PATH),
        "mapping": str(MAPPING_PATH),
        "independent_observations": int(len(independent)),
        "groups": int(independent["Group"].nunique()),
        "wavelengths_nm": [440, 250, 308],
        "time_points_per_wavelength": int(len(data.wavelengths["440"].time)),
        "time_range_min": [
            float(data.wavelengths["440"].time[0]),
            float(data.wavelengths["440"].time[-1]),
        ],
        "cross_validation": {
            "strategy": "Repeated provenance-aware StratifiedGroupKFold",
            "n_splits": CV_N_SPLITS,
            "n_repeats": CV_REPEATS,
            "random_state": RANDOM_STATE,
            "total_splits_per_preprocessing_modality": CV_N_SPLITS * CV_REPEATS,
            "grouping_rule": "Samples 39 and 40 are forced into the same fold because 308 nm Sample 40 is a deterministic 0.53-scaled copy of Sample 39.",
        },
        "provenance_constraints": [
            {
                "samples": [39, 40],
                "modalities_affected": ["308 nm", "Combined"],
                "relation": "Sample 40 = Sample 39 * 0.53",
            }
        ],
        "model": {
            "classifier": "PLS-DA",
            "components": 4,
            "pipeline": [
                "SimpleImputer(median)",
                "StandardScaler",
                "PLSRegression(n_components=4, scale=False)",
            ],
        },
        "dimensionality_reduction": "No PCA. PLS-DA itself provides the supervised latent representation used by the classifier.",
        "preprocessing_candidates": [
            "Raw",
            "SNV",
            "AsLS + SNV",
            "AsLS + L2",
        ],
        "modalities": ["440 nm", "250 nm", "308 nm", "Combined"],
        "asls_lambda": ASLS_LAMBDA,
        "asls_p": ASLS_P,
        "asls_iterations": ASLS_ITERATIONS,
        "descriptive_prediction_preprocessing": "AsLS + SNV",
        "scientific_status": (
            "Research / exploratory. Robust evaluation is descriptive for this pilot; "
            "no operational winner is selected."
        ),
        "outputs": sorted(p.name for p in OUTPUTS_DIR.glob("*")),
    }
    (OUTPUTS_DIR / "pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Pipeline finished successfully.")
    print(f"Outputs: {OUTPUTS_DIR}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
