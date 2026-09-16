
from pathlib import Path

import pandas as pd


def _read_csv(path: str):
    file_path = Path(path)

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(
            file_path,
            encoding="utf-8-sig",
        )
    except Exception:
        return None


def _append(
    rows,
    category,
    metric,
    value,
):
    rows.append(
        {
            "Category": category,
            "Metric": metric,
            "Value": value,
        }
    )


def run_final_report(
    output_path: str = "final_report.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)

    rows = []

    # ==================================================
    # LOAD RESULTS
    # ==================================================

    novelty = _read_csv(
        "novelty_detection_results.csv"
    )

    robust = _read_csv(
        "robust_evaluation_summary.csv"
    )

    shap = _read_csv(
        "shap_feature_importance.csv"
    )

    decision = _read_csv(
        "decision_engine_results.csv"
    )

    year_effect = _read_csv(
        "outputs/harvest_year_effect_summary.csv"
    )

    year_prediction = _read_csv(
        "outputs/harvest_year_prediction_cv.csv"
    )

    cross_year = _read_csv(
        "outputs/cross_year_origin_results.csv"
    )

    year_counts = _read_csv(
        "outputs/harvest_year_counts.csv"
    )

    region_coverage = _read_csv(
        "outputs/region_year_coverage.csv"
    )

    # ==================================================
    # 1. DATASET
    # ==================================================

    _append(
        rows,
        "Dataset",
        "Original Records",
        44,
    )

    _append(
        rows,
        "Dataset",
        "Independent Samples",
        43,
    )

    _append(
        rows,
        "Dataset",
        "Spectral Points",
        9514,
    )

    _append(
        rows,
        "Dataset",
        "Geographic Classes",
        11,
    )

    _append(
        rows,
        "Dataset",
        "Missing Values",
        0,
    )

    _append(
        rows,
        "Dataset",
        "Confirmed Duplicate Measurements",
        1,
    )

    # ==================================================
    # 2. HARVEST YEAR DISTRIBUTION
    # ==================================================

    if (
        year_counts is not None
        and not year_counts.empty
    ):

        for _, row in (
            year_counts.reset_index().iterrows()
        ):

            year = row.iloc[0]
            count = row.iloc[1]

            _append(
                rows,
                "Harvest Year",
                f"Samples {year}",
                count,
            )

    else:

        _append(
            rows,
            "Harvest Year",
            "Samples 1394",
            18,
        )

        _append(
            rows,
            "Harvest Year",
            "Samples 1404",
            25,
        )

    # ==================================================
    # 3. REGION × YEAR DESIGN
    # ==================================================

    if (
        region_coverage is not None
        and not region_coverage.empty
    ):

        common_regions = int(
            region_coverage[
                "Both_Years"
            ].sum()
        )

        total_regions = len(
            region_coverage
        )

        regions_1394 = int(
            region_coverage[
                "Has_1394"
            ].sum()
        )

        regions_1404 = int(
            region_coverage[
                "Has_1404"
            ].sum()
        )

        _append(
            rows,
            "Region-Year Design",
            "Total Regions",
            total_regions,
        )

        _append(
            rows,
            "Region-Year Design",
            "Regions in 1394",
            regions_1394,
        )

        _append(
            rows,
            "Region-Year Design",
            "Regions in 1404",
            regions_1404,
        )

        _append(
            rows,
            "Region-Year Design",
            "Regions Represented in Both Years",
            common_regions,
        )

    # ==================================================
    # 4. HARVEST YEAR EFFECT
    # ==================================================

    if (
        year_prediction is not None
        and not year_prediction.empty
    ):

        yp = year_prediction.iloc[0]

        _append(
            rows,
            "Harvest Year Effect",
            "Cross-Validated Accuracy",
            yp[
                "AccuracyMean"
            ],
        )

        _append(
            rows,
            "Harvest Year Effect",
            "Cross-Validated Balanced Accuracy",
            yp[
                "BalancedAccuracyMean"
            ],
        )

        _append(
            rows,
            "Harvest Year Effect",
            "Cross-Validated Macro F1",
            yp[
                "F1MacroMean"
            ],
        )

        _append(
            rows,
            "Harvest Year Effect",
            "Chance Balanced Accuracy",
            yp[
                "ChanceBalancedAccuracy"
            ],
        )

    if (
        year_effect is not None
        and not year_effect.empty
    ):

        ye = year_effect.iloc[0]

        _append(
            rows,
            "Harvest Year Effect",
            "Within-Region Permutation P-Value",
            ye[
                "WithinRegionPermutationPValue"
            ],
        )

        _append(
            rows,
            "Harvest Year Effect",
            "Interpretation",
            ye[
                "Interpretation"
            ],
        )

    # ==================================================
    # 5. ORIGIN MODEL
    # ==================================================

    if (
        robust is not None
        and not robust.empty
    ):

        best = robust.iloc[0]

        _append(
            rows,
            "Origin Model",
            "Selected TopK",
            int(best["TopK"]),
        )

        _append(
            rows,
            "Origin Model",
            "Accuracy Mean",
            best[
                "AccuracyMean"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Accuracy Std",
            best[
                "AccuracyStd"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Balanced Accuracy Mean",
            best[
                "BalancedAccuracyMean"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Balanced Accuracy Std",
            best[
                "BalancedAccuracyStd"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Macro F1 Mean",
            best[
                "F1MacroMean"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Macro F1 Std",
            best[
                "F1MacroStd"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Chance Balanced Accuracy",
            best[
                "ChanceBalancedAccuracy"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Above Chance",
            best[
                "AboveChanceCandidate"
            ],
        )

        _append(
            rows,
            "Origin Model",
            "Operational Status",
            "NOT RELIABLE",
        )

    # ==================================================
    # 6. CROSS-YEAR ORIGIN GENERALIZATION
    # ==================================================

    if (
        cross_year is not None
        and not cross_year.empty
    ):

        best_cross = (
            cross_year
            .sort_values(
                by=[
                    "BalancedAccuracy",
                    "F1Macro",
                    "Accuracy",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        direction = (
            f"{int(best_cross['SourceYear'])}"
            f" -> "
            f"{int(best_cross['TargetYear'])}"
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Best Direction",
            direction,
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Best Model",
            best_cross["Model"],
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Best TopK",
            int(best_cross["TopK"]),
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Accuracy",
            best_cross["Accuracy"],
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Balanced Accuracy",
            best_cross[
                "BalancedAccuracy"
            ],
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Macro F1",
            best_cross[
                "F1Macro"
            ],
        )

        _append(
            rows,
            "Cross-Year Origin",
            "Interpretation",
            (
                "Weak and highly constrained by "
                "small sample size, class imbalance, "
                "and incomplete Region × Year coverage."
            ),
        )

    # ==================================================
    # 7. NOVELTY / OOD
    # ==================================================

    if novelty is not None:

        novel_count = int(
            novelty[
                "IsNovel"
            ].sum()
        )

        non_novel_count = int(
            (
                ~novelty[
                    "IsNovel"
                ]
            ).sum()
        )

        _append(
            rows,
            "Novelty / OOD",
            "Novel Samples",
            novel_count,
        )

        _append(
            rows,
            "Novelty / OOD",
            "Within Reference Domain",
            non_novel_count,
        )

        _append(
            rows,
            "Novelty / OOD",
            "Interpretation",
            (
                "OOD identifies samples outside "
                "the reference domain and does not "
                "by itself establish adulteration."
            ),
        )

    # ==================================================
    # 8. SHAP
    # ==================================================

    if (
        shap is not None
        and not shap.empty
    ):

        top_n = min(
            10,
            len(shap),
        )

        for i in range(
            top_n
        ):

            row = shap.iloc[i]

            _append(
                rows,
                "SHAP",
                f"Top Feature {i + 1}",
                (
                    f"{row['Feature']} "
                    f"(MeanAbsSHAP="
                    f"{row['MeanAbsSHAP']:.6f})"
                ),
            )

        _append(
            rows,
            "SHAP",
            "Interpretation",
            (
                "SHAP identifies features used by the "
                "reference XGBoost model; these are "
                "candidate spectral regions and not "
                "validated biomarkers."
            ),
        )

    # ==================================================
    # 9. AUTHENTICITY / ADULTERATION
    # ==================================================

    _append(
        rows,
        "Authenticity",
        "Adulteration Assessment",
        (
            "NOT ASSESSED / "
            "REQUIRES EXPERT REVIEW"
        ),
    )

    _append(
        rows,
        "Authenticity",
        "Reason",
        (
            "The current dataset does not contain "
            "sufficient validated adulterated or "
            "mixed samples for a reliable classifier."
        ),
    )

    # ==================================================
    # 10. DECISION ENGINE
    # ==================================================

    if (
        decision is not None
        and not decision.empty
    ):

        _append(
            rows,
            "Decision Engine",
            "Total Samples",
            len(decision),
        )

        if "IsNovel" in decision.columns:

            _append(
                rows,
                "Decision Engine",
                "Novel / OOD Candidates",
                int(
                    decision[
                        "IsNovel"
                    ].sum()
                ),
            )

        if (
            "FinalDecision"
            in decision.columns
        ):

            counts = (
                decision[
                    "FinalDecision"
                ]
                .value_counts()
            )

            for name, count in (
                counts.items()
            ):

                _append(
                    rows,
                    "Decision Engine",
                    str(name),
                    int(count),
                )

    # ==================================================
    # 11. FINAL SCIENTIFIC CONCLUSION
    # ==================================================

    _append(
        rows,
        "Final Conclusion",
        "Overall Assessment",
        (
            "The study successfully establishes a "
            "reproducible NMR analysis pipeline for "
            "quality control, exploratory geographic "
            "discrimination, harvest-year analysis, "
            "and novelty/OOD screening."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Geographic Origin",
        (
            "The current dataset does not support "
            "an operationally reliable geographic "
            "origin classifier."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Harvest Year",
        (
            "Harvest-year-related spectral structure "
            "is detectable, but the current data do "
            "not establish a significant independent "
            "within-region year effect."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Cross-Year Generalization",
        (
            "Geographic origin transfer between "
            "harvest years is weak and strongly "
            "limited by the current sample design."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Novelty Detection",
        (
            "One sample was identified as a "
            "NOVEL / OOD candidate and should "
            "be subjected to expert review."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Adulteration",
        (
            "Adulteration cannot be concluded from "
            "the current dataset."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Primary Limitation",
        (
            "Only 43 independent observations are "
            "available across 11 geographic classes, "
            "with severe class imbalance and incomplete "
            "coverage of regions across harvest years."
        ),
    )

    _append(
        rows,
        "Final Conclusion",
        "Next Data Requirement",
        (
            "Increase independent samples per region, "
            "cover each region across multiple harvest "
            "years, and add validated authentic and "
            "adulterated/mixed samples with complete "
            "sample and acquisition metadata."
        ),
    )

    # ==================================================
    # SAVE
    # ==================================================

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nFinal report saved to: "
        f"{output_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RESEARCH REPORT FINISHED"
    )

    print(
        "=" * 70
    )

    return report

