from pathlib import Path
import pandas as pd
import streamlit as st

from components.common import apply_page_style, get_dataset_info


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"

YEAR_PREDICTION_PATH = OUTPUTS_DIR / "harvest_year_prediction_cv.csv"
YEAR_EFFECT_PATH = OUTPUTS_DIR / "within_region_year_effect.csv"
YEAR_EFFECT_SUMMARY_PATH = OUTPUTS_DIR / "harvest_year_effect_summary.csv"

REGION_DISTRIBUTION_PATH = (
    OUTPUTS_DIR
    / "region_year_analysis"
    / "region_distribution.csv"
)
REGIONAL_MODEL_RESULTS_PATH = (
    OUTPUTS_DIR
    / "region_year_analysis"
    / "regional_model_results.csv"
)
REGION_YEAR_SIMILARITY_PATH = (
    OUTPUTS_DIR
    / "region_year_analysis"
    / "region_year_similarity.csv"
)
REGION_YEAR_PAIRWISE_PATH = (
    OUTPUTS_DIR
    / "region_year_analysis"
    / "region_year_pairwise_distances.csv"
)
REGION_YEAR_SUMMARY_PATH = (
    OUTPUTS_DIR
    / "region_year_analysis"
    / "analysis_summary.csv"
)

PCA_SCORES_PLOT = BASE_DIR / "pca_scores.png"
PCA_SCREE_PLOT = BASE_DIR / "pca_scree_plot.png"
YEAR_PCA_PLOT = OUTPUTS_DIR / "year_pca_plot.png"
YEAR_REGION_PCA_PLOT = OUTPUTS_DIR / "year_region_pca_plot.png"


apply_page_style()


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fmt(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


dataset = get_dataset_info()

st.title("📅 تحلیل سال برداشت زعفران")

st.caption(
    "PCA، Cross-Validated Year Prediction، Within-Region "
    "Effect و تحلیل تخصصی Region × Year"
)

st.divider()


# ============================================================
# YEAR OVERVIEW
# ============================================================

years = dataset["years"]
year_counts = dataset["year_counts"]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "سال‌های موجود",
        " / ".join(str(y) for y in years) if years else "—",
    )

with c2:
    st.metric(
        "نمونه‌های مستقل",
        dataset["independent"],
    )

with c3:
    region_year = dataset["region_year"]
    common_regions = (
        int((((region_year > 0).sum(axis=1)) >= 2).sum())
        if not region_year.empty
        else 0
    )
    st.metric(
        "مناطق دارای هر دو سال",
        common_regions,
    )


st.divider()


# ============================================================
# PCA
# ============================================================

st.subheader("📈 PCA")

p1, p2 = st.columns(2)

with p1:
    if PCA_SCORES_PLOT.exists():
        st.image(
            str(PCA_SCORES_PLOT),
            caption="PCA Scores",
            width="stretch",
        )
    else:
        st.info("pca_scores.png موجود نیست.")

with p2:
    if PCA_SCREE_PLOT.exists():
        st.image(
            str(PCA_SCREE_PLOT),
            caption="PCA Scree Plot",
            width="stretch",
        )
    else:
        st.info("pca_scree_plot.png موجود نیست.")


st.divider()

st.subheader("PCA اختصاصی سال برداشت")

p1, p2 = st.columns(2)

with p1:
    if YEAR_PCA_PLOT.exists():
        st.image(
            str(YEAR_PCA_PLOT),
            caption="Harvest Year PCA",
            width="stretch",
        )
    else:
        st.info("year_pca_plot.png موجود نیست.")

with p2:
    if YEAR_REGION_PCA_PLOT.exists():
        st.image(
            str(YEAR_REGION_PCA_PLOT),
            caption="Region × Harvest Year PCA",
            width="stretch",
        )
    else:
        st.info("year_region_pca_plot.png موجود نیست.")


st.divider()


# ============================================================
# YEAR DISTRIBUTION
# ============================================================

st.subheader("توزیع سال برداشت")

if year_counts.empty:
    st.info("اطلاعات سال برداشت موجود نیست.")
else:
    chart = year_counts[
        ["HarvestYear", "Samples"]
    ].copy()

    chart["HarvestYear"] = (
        chart["HarvestYear"]
        .astype(int)
        .astype(str)
    )

    st.bar_chart(
        chart.set_index("HarvestYear")["Samples"]
    )

    st.dataframe(
        year_counts,
        hide_index=True,
        use_container_width=True,
    )


st.divider()


# ============================================================
# CROSS-VALIDATED YEAR PREDICTION
# ============================================================

st.subheader("🎯 Cross-Validated Harvest-Year Prediction")

year_prediction = read_csv(
    str(YEAR_PREDICTION_PATH)
)

if year_prediction.empty:
    st.info(
        "harvest_year_prediction_cv.csv موجود نیست."
    )
else:
    st.dataframe(
        year_prediction,
        hide_index=True,
        use_container_width=True,
    )

    numeric = year_prediction.copy()

    metrics_map = {
        "Accuracy": [
            "Accuracy",
            "AccuracyMean",
        ],
        "Balanced Accuracy": [
            "BalancedAccuracy",
            "BalancedAccuracyMean",
        ],
        "Macro F1": [
            "MacroF1",
            "F1MacroMean",
        ],
    }

    values = {}

    for label, candidates in metrics_map.items():
        found = next(
            (
                col
                for col in candidates
                if col in numeric.columns
            ),
            None,
        )
        if found is not None:
            series = pd.to_numeric(
                numeric[found],
                errors="coerce",
            ).dropna()

            if not series.empty:
                values[label] = series.mean()

    if values:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Accuracy",
                fmt(values.get("Accuracy")),
            )
        with c2:
            st.metric(
                "Balanced Accuracy",
                fmt(values.get("Balanced Accuracy")),
            )
        with c3:
            st.metric(
                "Macro F1",
                fmt(values.get("Macro F1")),
            )


st.divider()


# ============================================================
# WITHIN-REGION YEAR EFFECT
# ============================================================

st.subheader("🔬 Within-Region Harvest-Year Effect")

year_effect = read_csv(
    str(YEAR_EFFECT_PATH)
)

if year_effect.empty:
    st.info(
        "within_region_year_effect.csv موجود نیست."
    )
else:
    st.dataframe(
        year_effect,
        hide_index=True,
        use_container_width=True,
    )

year_summary = read_csv(
    str(YEAR_EFFECT_SUMMARY_PATH)
)

if not year_summary.empty:
    st.markdown("### Year Effect Summary")
    st.dataframe(
        year_summary,
        hide_index=True,
        use_container_width=True,
    )


st.warning(
    """
    در آزمون کنترل‌شده درون‌منطقه‌ای، نتیجه فعلی به‌تنهایی
    برای اثبات یک اثر آماری قطعی سال برداشت کافی نیست.
    """
)


st.divider()


# ============================================================
# REGION × YEAR SPECIAL ANALYSIS
# ============================================================

st.subheader("🌍 تحلیل تخصصی Region × Year")

summary = read_csv(
    str(REGION_YEAR_SUMMARY_PATH)
)
region_distribution = read_csv(
    str(REGION_DISTRIBUTION_PATH)
)
regional_results = read_csv(
    str(REGIONAL_MODEL_RESULTS_PATH)
)
similarity = read_csv(
    str(REGION_YEAR_SIMILARITY_PATH)
)
pairwise = read_csv(
    str(REGION_YEAR_PAIRWISE_PATH)
)

region_ready = any(
    not df.empty
    for df in [
        summary,
        region_distribution,
        regional_results,
        similarity,
    ]
)

if not region_ready:
    st.info(
        "تحلیل تخصصی منطقه و سال هنوز اجرا نشده است."
    )
else:
    s = (
        summary.iloc[0]
        if not summary.empty
        else pd.Series(dtype=object)
    )

    independent = (
        s.get("IndependentSamples")
        if "IndependentSamples" in summary.columns
        else dataset["independent"]
    )

    selected_regions = (
        s.get("SelectedRegions")
        if "SelectedRegions" in summary.columns
        else len(region_distribution)
        if not region_distribution.empty
        else "—"
    )

    common_regions = (
        s.get("CommonRegionsBothYears")
        if "CommonRegionsBothYears" in summary.columns
        else common_regions
    )

    feature_count = (
        s.get("FeatureCount")
        if "FeatureCount" in summary.columns
        else 401
    )

    feature_source = (
        s.get("FeatureSource")
        if "FeatureSource" in summary.columns
        else "engineered_features.csv"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "نمونه‌های مستقل",
            str(int(float(independent)))
            if pd.notna(independent)
            else "—",
        )

    with c2:
        st.metric(
            "مناطق منتخب",
            str(int(float(selected_regions)))
            if pd.notna(selected_regions)
            else "—",
        )

    with c3:
        st.metric(
            "مناطق دارای هر دو سال",
            str(int(float(common_regions)))
            if pd.notna(common_regions)
            else "—",
        )

    with c4:
        st.metric(
            "ویژگی‌های مدل منطقه‌ای",
            str(int(float(feature_count)))
            if pd.notna(feature_count)
            else "—",
        )

    with c5:
        st.metric(
            "نقاط طیفی",
            dataset["spectral_points"],
        )

    st.caption(
        f"منبع ویژگی‌ها: **{feature_source}**"
    )


    # --------------------------------------------------------
    # Regional model results
    # --------------------------------------------------------

    if not region_distribution.empty:
        st.markdown("### مناطق منتخب")
        st.dataframe(
            region_distribution,
            hide_index=True,
            use_container_width=True,
        )

    if not regional_results.empty:
        rr = regional_results.copy()

        if "Region" not in rr.columns:
            for candidate in [
                "group",
                "Group",
                "region",
            ]:
                if candidate in rr.columns:
                    rr["Region"] = (
                        rr[candidate].astype(str)
                    )
                    break

        if "RegionLabel" not in rr.columns:
            rr["RegionLabel"] = (
                rr["Region"]
                if "Region" in rr.columns
                else "—"
            )

        numeric_cols = [
            "Samples",
            "PositiveSamples",
            "TopK",
            "AccuracyMean",
            "AccuracyStd",
            "BalancedAccuracyMean",
            "BalancedAccuracyStd",
            "MacroF1Mean",
            "MacroF1Std",
        ]

        for col in numeric_cols:
            if col in rr.columns:
                rr[col] = pd.to_numeric(
                    rr[col],
                    errors="coerce",
                )

        if {
            "Region",
            "BalancedAccuracyMean",
        }.issubset(rr.columns):

            best_per_region = (
                rr.sort_values(
                    [
                        "Region",
                        "BalancedAccuracyMean",
                    ],
                    ascending=[True, False],
                )
                .groupby(
                    "Region",
                    as_index=False,
                )
                .head(1)
            )

            st.markdown(
                "### بهترین نتیجه گزارش‌شده برای هر منطقه"
            )

            cols = [
                "Region",
                "RegionLabel",
                "Samples",
                "PositiveSamples",
                "Model",
                "TopK",
                "AccuracyMean",
                "BalancedAccuracyMean",
                "MacroF1Mean",
            ]

            cols = [
                c for c in cols
                if c in best_per_region.columns
            ]

            st.dataframe(
                best_per_region[cols].round(4),
                hide_index=True,
                use_container_width=True,
            )

            if {
                "RegionLabel",
                "BalancedAccuracyMean",
            }.issubset(best_per_region.columns):
                chart = (
                    best_per_region[
                        [
                            "RegionLabel",
                            "BalancedAccuracyMean",
                        ]
                    ]
                    .dropna()
                    .set_index("RegionLabel")
                )

                st.markdown(
                    "### Balanced Accuracy مناطق منتخب"
                )
                if not chart.empty:
                    st.bar_chart(chart)

            if {
                "RegionLabel",
                "BalancedAccuracyMean",
                "BalancedAccuracyStd",
            }.issubset(
                best_per_region.columns
            ):
                uncertainty = (
                    best_per_region[
                        [
                            "RegionLabel",
                            "BalancedAccuracyMean",
                            "BalancedAccuracyStd",
                        ]
                    ]
                    .rename(
                        columns={
                            "RegionLabel": "منطقه",
                            "BalancedAccuracyMean": "میانگین BA",
                            "BalancedAccuracyStd": "انحراف معیار BA",
                        }
                    )
                )

                st.markdown(
                    "### عدم‌قطعیت Balanced Accuracy"
                )
                st.dataframe(
                    uncertainty.round(4),
                    hide_index=True,
                    use_container_width=True,
                )

        st.markdown(
            "### تمام نتایج مدل‌های منطقه‌ای"
        )

        display_cols = [
            c
            for c in [
                "Region",
                "RegionLabel",
                "Samples",
                "PositiveSamples",
                "Model",
                "TopK",
                "AccuracyMean",
                "AccuracyStd",
                "BalancedAccuracyMean",
                "BalancedAccuracyStd",
                "MacroF1Mean",
                "MacroF1Std",
            ]
            if c in rr.columns
        ]

        if display_cols:
            st.dataframe(
                rr[display_cols].round(4),
                hide_index=True,
                use_container_width=True,
            )

    st.info(
        """
        مدل‌های منطقه‌ای اختصاصی در این بخش ماهیت اکتشافی دارند.
        اعداد این بخش به‌تنهایی اثبات قطعی منشأ جغرافیایی نیستند.
        """
    )


    # --------------------------------------------------------
    # Similarity across years
    # --------------------------------------------------------

    if not similarity.empty:
        st.markdown(
            "### پایداری الگوی منطقه‌ای بین ۱۳۹۴ و ۱۴۰۴"
        )

        sim = similarity.copy()

        aliases = {
            "SameRegionDistanceMean": [
                "SameRegionDistanceMean",
                "MeanSameRegionDistance",
            ],
            "OtherRegionDistanceMean": [
                "OtherRegionDistanceMean",
                "MeanOtherRegionDistance",
            ],
            "SameToOtherDistanceRatio": [
                "SameToOtherDistanceRatio",
            ],
            "NearestSameRegionFraction": [
                "NearestSameRegionFraction",
            ],
            "PermutationPValue": [
                "PermutationPValue",
            ],
        }

        for canonical, candidates in aliases.items():
            if canonical not in sim.columns:
                found = next(
                    (
                        c
                        for c in candidates
                        if c in sim.columns
                    ),
                    None,
                )
                if found:
                    sim[canonical] = sim[found]

        if "Region" not in sim.columns:
            for candidate in [
                "group",
                "Group",
            ]:
                if candidate in sim.columns:
                    sim["Region"] = (
                        sim[candidate].astype(str)
                    )
                    break

        if "RegionLabel" not in sim.columns:
            sim["RegionLabel"] = (
                sim["Region"]
                if "Region" in sim.columns
                else "—"
            )

        for col in [
            "SameRegionDistanceMean",
            "OtherRegionDistanceMean",
            "SameToOtherDistanceRatio",
            "NearestSameRegionFraction",
            "PermutationPValue",
        ]:
            if col in sim.columns:
                sim[col] = pd.to_numeric(
                    sim[col],
                    errors="coerce",
                )

        overall = None

        if "Region" in sim.columns:
            rows = sim[
                sim["Region"]
                .astype(str)
                .str.upper()
                == "ALL"
            ]

            if not rows.empty:
                overall = rows.iloc[0]

        if overall is not None:
            same_mean = overall.get(
                "SameRegionDistanceMean",
                float("nan"),
            )
            other_mean = overall.get(
                "OtherRegionDistanceMean",
                float("nan"),
            )
            ratio = overall.get(
                "SameToOtherDistanceRatio",
                float("nan"),
            )
            nearest = overall.get(
                "NearestSameRegionFraction",
                float("nan"),
            )
            p_value = overall.get(
                "PermutationPValue",
                float("nan"),
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "میانگین فاصله همان منطقه",
                    fmt(same_mean, 2),
                )

            with c2:
                st.metric(
                    "میانگین فاصله سایر مناطق",
                    fmt(other_mean, 2),
                )

            with c3:
                st.metric(
                    "نسبت Same / Other",
                    fmt(ratio),
                )

            with c4:
                st.metric(
                    "Nearest Same-Region",
                    (
                        f"{fmt(nearest * 100, 1)}%"
                        if pd.notna(nearest)
                        else "—"
                    ),
                )

            with c5:
                st.metric(
                    "Permutation p-value",
                    fmt(p_value),
                )

            if (
                pd.notna(ratio)
                and pd.notna(nearest)
                and pd.notna(p_value)
            ):
                if (
                    ratio < 1
                    and nearest > 0.5
                    and p_value < 0.05
                ):
                    st.success(
                        "این سه معیار هم‌زمان با پایداری "
                        "الگوی منطقه‌ای بین دو سال سازگارند."
                    )
                else:
                    st.warning(
                        "شواهد کافی برای تأیید پایداری "
                        "یک امضای منطقه‌ای بین دو سال مشاهده نشد."
                    )

        sim_display = sim.copy().rename(
            columns={
                "Region": "کد منطقه",
                "RegionLabel": "منطقه",
                "N_1394": "نمونه ۱۳۹۴",
                "N_1404": "نمونه ۱۴۰۴",
                "SameRegionDistanceMean": "فاصله همان منطقه",
                "OtherRegionDistanceMean": "فاصله سایر مناطق",
                "SameToOtherDistanceRatio": "نسبت Same/Other",
                "NearestSameRegionFraction": "نزدیک‌ترین هم‌منطقه",
                "PermutationPValue": "p-value",
                "EvidenceRatioBelow1": "Ratio < 1",
                "EvidenceNearestAbove05": "Nearest > 0.5",
            }
        )

        for col in [
            "فاصله همان منطقه",
            "فاصله سایر مناطق",
            "نسبت Same/Other",
            "نزدیک‌ترین هم‌منطقه",
            "p-value",
        ]:
            if col in sim_display.columns:
                sim_display[col] = pd.to_numeric(
                    sim_display[col],
                    errors="coerce",
                ).round(4)

        st.dataframe(
            sim_display,
            hide_index=True,
            use_container_width=True,
        )

        if {
            "Region",
            "RegionLabel",
            "SameRegionDistanceMean",
            "OtherRegionDistanceMean",
        }.issubset(sim.columns):
            chart = (
                sim[
                    sim["Region"]
                    .astype(str)
                    .str.upper()
                    != "ALL"
                ][
                    [
                        "RegionLabel",
                        "SameRegionDistanceMean",
                        "OtherRegionDistanceMean",
                    ]
                ]
                .set_index("RegionLabel")
            )

            st.markdown(
                "### فاصله همان منطقه در برابر سایر مناطق"
            )
            if not chart.empty:
                st.bar_chart(chart)

        if {
            "Region",
            "RegionLabel",
            "NearestSameRegionFraction",
        }.issubset(sim.columns):
            chart = (
                sim[
                    sim["Region"]
                    .astype(str)
                    .str.upper()
                    != "ALL"
                ][
                    [
                        "RegionLabel",
                        "NearestSameRegionFraction",
                    ]
                ]
                .set_index("RegionLabel")
            )

            st.markdown(
                "### سهم نزدیک‌ترین نمونه از همان منطقه"
            )
            if not chart.empty:
                st.bar_chart(chart)

    if not pairwise.empty:
        with st.expander(
            "▶ مشاهده تمام فاصله‌های زوجی بین ۱۳۹۴ و ۱۴۰۴"
        ):
            st.dataframe(
                pairwise,
                hide_index=True,
                use_container_width=True,
            )


st.divider()


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

st.subheader("🧠 تفسیر علمی")

st.success(
    """
    PCA، Cross-Validated Year Prediction و تحلیل‌های
    Region × Year در Pipeline اجرا شده‌اند.
    """
)

st.warning(
    """
    Cross-Validated year prediction می‌تواند نشان‌دهنده
    وجود ساختار مرتبط با سال باشد، اما آزمون Within-Region
    برای تصمیم‌گیری نهایی باید در کنار حجم نمونه، طراحی آزمایش
    و اعتبارسنجی مستقل تفسیر شود.
    """
)
