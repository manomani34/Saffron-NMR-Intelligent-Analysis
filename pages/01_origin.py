from pathlib import Path
import pandas as pd
import streamlit as st

from components.common import apply_page_style, get_dataset_info


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"

ROBUST_SUMMARY_PATH = BASE_DIR / "robust_evaluation_summary.csv"
PREDICTIONS_PATH = BASE_DIR / "sample_predictions.csv"
NOVELTY_PATH = BASE_DIR / "novelty_detection_results.csv"
DECISION_PATH = BASE_DIR / "decision_engine_results.csv"
SHAP_PATH = BASE_DIR / "shap_feature_importance.csv"
CROSS_YEAR_PATH = OUTPUTS_DIR / "cross_year_origin_results.csv"
RUN_OUTPUT_PATH = BASE_DIR / "run_output.txt"
FINAL_REPORT_PATH = BASE_DIR / "final_report.csv"
README_PATH = BASE_DIR / "README.md"
CONCLUSION_PATH = BASE_DIR / "docs" / "conclusion_fa.md"


apply_page_style()


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_text(path_string: str) -> str:
    path = Path(path_string)

    if not path.exists():
        return ""

    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return ""


def fmt(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def fmt_pm(mean, std):
    if pd.isna(mean):
        return "—"

    if pd.isna(std):
        return fmt(mean)

    return f"{fmt(mean)} ± {fmt(std)}"


def normalize_bool(value):
    return (
        str(value).strip().lower()
        in {"true", "1", "yes"}
    )


def build_confusion_matrix(predictions):
    required = {
        "ActualGroup",
        "PredictedGroup",
    }

    if (
        predictions.empty
        or not required.issubset(predictions.columns)
    ):
        return pd.DataFrame()

    actual = predictions["ActualGroup"].astype(str)
    predicted = predictions["PredictedGroup"].astype(str)

    labels = sorted(
        set(actual) | set(predicted)
    )

    matrix = pd.crosstab(
        actual,
        predicted,
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )

    return matrix.reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )


def build_error_analysis(predictions):
    required = {
        "ActualGroup",
        "PredictedGroup",
    }

    if (
        predictions.empty
        or not required.issubset(predictions.columns)
    ):
        return pd.DataFrame(), pd.DataFrame()

    df = predictions.copy()

    df["ActualGroup"] = (
        df["ActualGroup"].astype(str)
    )

    df["PredictedGroup"] = (
        df["PredictedGroup"].astype(str)
    )

    df["CorrectCalc"] = (
        df["ActualGroup"]
        == df["PredictedGroup"]
    )

    class_summary = (
        df.groupby("ActualGroup")
        .agg(
            Samples=("ActualGroup", "size"),
            Correct=("CorrectCalc", "sum"),
        )
        .reset_index()
    )

    class_summary["Errors"] = (
        class_summary["Samples"]
        - class_summary["Correct"]
    )

    class_summary["Accuracy"] = (
        class_summary["Correct"]
        / class_summary["Samples"]
    )

    class_summary["ErrorRate"] = (
        1
        - class_summary["Accuracy"]
    )

    class_summary = (
        class_summary.sort_values(
            ["ErrorRate", "Samples"],
            ascending=[False, False],
        )
    )

    errors = df[
        ~df["CorrectCalc"]
    ]

    if errors.empty:
        confusion_summary = pd.DataFrame()
    else:
        confusion_summary = (
            errors.groupby(
                [
                    "ActualGroup",
                    "PredictedGroup",
                ]
            )
            .size()
            .reset_index(
                name="Errors"
            )
            .sort_values(
                "Errors",
                ascending=False,
            )
        )

    return (
        class_summary,
        confusion_summary,
    )


def build_confidence_distribution(predictions):
    if (
        predictions.empty
        or "PredictionConfidence"
        not in predictions.columns
    ):
        return pd.DataFrame()

    confidence = pd.to_numeric(
        predictions[
            "PredictionConfidence"
        ],
        errors="coerce",
    ).dropna()

    if confidence.empty:
        return pd.DataFrame()

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    labels = [
        "0.0–0.1",
        "0.1–0.2",
        "0.2–0.3",
        "0.3–0.4",
        "0.4–0.5",
        "0.5–0.6",
        "0.6–0.7",
        "0.7–0.8",
        "0.8–0.9",
        "0.9–1.0",
    ]

    categories = pd.cut(
        confidence,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return (
        categories.value_counts()
        .sort_index()
        .rename("Samples")
        .to_frame()
    )


# ============================================================
# DATASET
# ============================================================

dataset = get_dataset_info()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 تعیین منشأ جغرافیایی زعفران"
)

st.caption(
    "تحلیل مدل‌های طبقه‌بندی منشأ، ارزیابی Robust، "
    "Cross-Year، Novelty/OOD و بررسی نمونه"
)

st.divider()


# ============================================================
# DATASET SNAPSHOT
# ============================================================

st.subheader("وضعیت Dataset")

a, b, c, d = st.columns(4)

with a:
    st.metric(
        "رکوردهای اولیه",
        dataset["records"],
    )

with b:
    st.metric(
        "نمونه‌های مستقل",
        dataset["independent"],
    )

with c:
    st.metric(
        "مناطق",
        dataset["groups"],
    )

with d:
    st.metric(
        "نقاط طیفی",
        f"{dataset['spectral_points']:,}",
    )


st.divider()


# ============================================================
# ROBUST MODEL EVALUATION
# ============================================================

st.subheader(
    "📊 ارزیابی Robust مدل منشأ"
)

robust = read_csv(
    str(ROBUST_SUMMARY_PATH)
)

if robust.empty:
    st.warning(
        "robust_evaluation_summary.csv موجود نیست."
    )

else:
    required = {
        "TopK",
        "AccuracyMean",
        "AccuracyStd",
        "BalancedAccuracyMean",
        "BalancedAccuracyStd",
        "F1MacroMean",
        "F1MacroStd",
    }

    if required.issubset(
        robust.columns
    ):
        table = robust.copy()

        table["Accuracy"] = table.apply(
            lambda row: fmt_pm(
                row["AccuracyMean"],
                row["AccuracyStd"],
            ),
            axis=1,
        )

        table["Balanced Accuracy"] = (
            table.apply(
                lambda row: fmt_pm(
                    row["BalancedAccuracyMean"],
                    row["BalancedAccuracyStd"],
                ),
                axis=1,
            )
        )

        table["Macro F1"] = table.apply(
            lambda row: fmt_pm(
                row["F1MacroMean"],
                row["F1MacroStd"],
            ),
            axis=1,
        )

        display_columns = [
            "TopK",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
        ]

        st.dataframe(
            table[display_columns],
            hide_index=True,
            use_container_width=True,
        )

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        # Descriptive maximum BA candidate only.
        numeric_ba = pd.to_numeric(
            robust["BalancedAccuracyMean"],
            errors="coerce",
        )

        if numeric_ba.notna().any():
            row = robust.loc[
                numeric_ba.idxmax()
            ]

            with c1:
                st.metric(
                    "TopK",
                    str(
                        int(
                            float(
                                row["TopK"]
                            )
                        )
                    ),
                )

            with c2:
                st.metric(
                    "Accuracy",
                    fmt(
                        row["AccuracyMean"]
                    ),
                )

            with c3:
                st.metric(
                    "Balanced Accuracy",
                    fmt(
                        row[
                            "BalancedAccuracyMean"
                        ]
                    ),
                )

            with c4:
                st.metric(
                    "Macro F1",
                    fmt(
                        row["F1MacroMean"]
                    ),
                )

            with c5:
                chance = (
                    row[
                        "ChanceBalancedAccuracy"
                    ]
                    if
                    "ChanceBalancedAccuracy"
                    in robust.columns
                    else None
                )

                st.metric(
                    "Chance BA",
                    fmt(chance),
                )

        st.markdown(
            "### Balanced Accuracy by TopK"
        )

        ba_chart = robust[
            [
                "TopK",
                "BalancedAccuracyMean",
            ]
        ].copy()

        ba_chart["TopK"] = (
            ba_chart["TopK"]
            .astype(int)
            .astype(str)
        )

        st.bar_chart(
            ba_chart.set_index(
                "TopK"
            )[
                "BalancedAccuracyMean"
            ]
        )

        st.markdown(
            "### Macro F1 by TopK"
        )

        f1_chart = robust[
            [
                "TopK",
                "F1MacroMean",
            ]
        ].copy()

        f1_chart["TopK"] = (
            f1_chart["TopK"]
            .astype(int)
            .astype(str)
        )

        st.bar_chart(
            f1_chart.set_index(
                "TopK"
            )[
                "F1MacroMean"
            ]
        )

    st.warning(
        """
        این نتایج مربوط به ارزیابی Robust فعلی هستند.
        با توجه به حجم و عدم‌توازن Dataset، مدل منشأ فعلی
        برای استفاده عملیاتی قابل اتکا نیست.
        """
    )


st.divider()


# ============================================================
# PREDICTIONS / DECISION DATA
# ============================================================

predictions = read_csv(
    str(PREDICTIONS_PATH)
)

novelty = read_csv(
    str(NOVELTY_PATH)
)

# FIX:
# Decision Engine results must also be loaded
# before being used in Sample Explorer.
decision = read_csv(
    str(DECISION_PATH)
)


# ============================================================
# CONFUSION / ERROR / CONFIDENCE
# ============================================================

st.subheader(
    "🔎 Confusion Matrix"
)

confusion = build_confusion_matrix(
    predictions
)

if confusion.empty:
    st.info(
        "Confusion Matrix قابل محاسبه نیست."
    )
else:
    st.dataframe(
        confusion,
        use_container_width=True,
    )


st.subheader(
    "⚠️ Error Analysis"
)

class_errors, confusion_errors = (
    build_error_analysis(
        predictions
    )
)

if class_errors.empty:
    st.info(
        "Error Analysis در دسترس نیست."
    )
else:
    display_errors = (
        class_errors.copy()
    )

    display_errors["Accuracy"] = (
        display_errors["Accuracy"]
        .map(fmt)
    )

    display_errors["ErrorRate"] = (
        display_errors["ErrorRate"]
        .map(fmt)
    )

    st.dataframe(
        display_errors,
        hide_index=True,
        use_container_width=True,
    )


if not confusion_errors.empty:
    st.markdown(
        "#### پرتکرارترین Confusionها"
    )

    st.dataframe(
        confusion_errors.head(15),
        hide_index=True,
        use_container_width=True,
    )


st.subheader(
    "🎯 Prediction Confidence"
)

confidence_dist = (
    build_confidence_distribution(
        predictions
    )
)

if confidence_dist.empty:
    st.info(
        "اطلاعات Confidence موجود نیست."
    )
else:
    st.bar_chart(
        confidence_dist["Samples"]
    )

    confidence_values = pd.to_numeric(
        predictions[
            "PredictionConfidence"
        ],
        errors="coerce",
    ).dropna()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Mean Confidence",
            fmt(
                confidence_values.mean()
            ),
        )

    with c2:
        st.metric(
            "Maximum Confidence",
            fmt(
                confidence_values.max()
            ),
        )

    with c3:
        st.metric(
            "Minimum Confidence",
            fmt(
                confidence_values.min()
            ),
        )


st.divider()


# ============================================================
# CROSS YEAR + OOD
# ============================================================

st.subheader(
    "🔁 Cross-Year Generalization"
)

cross_year = read_csv(
    str(CROSS_YEAR_PATH)
)

if cross_year.empty:
    st.info(
        "نتایج Cross-Year موجود نیست."
    )
else:
    st.dataframe(
        cross_year,
        hide_index=True,
        use_container_width=True,
    )

    if (
        "BalancedAccuracy"
        in cross_year.columns
    ):
        ba_values = pd.to_numeric(
            cross_year[
                "BalancedAccuracy"
            ],
            errors="coerce",
        )

        if ba_values.notna().any():
            row = cross_year.loc[
                ba_values.idxmax()
            ]

            st.info(
                f"""
                Cross-Year Candidate:
                **{int(row['SourceYear'])} → {int(row['TargetYear'])}**

                Model: **{row['Model']}**

                TopK: **{int(row['TopK'])}**

                Balanced Accuracy:
                **{fmt(row['BalancedAccuracy'])}**

                این نتیجه آزمون تعمیم بین‌سال است و
                عملکرد عملیاتی نهایی محسوب نمی‌شود.
                """
            )


st.markdown(
    "### Novelty / OOD"
)

if novelty.empty:
    st.info(
        "نتایج Novelty/OOD موجود نیست."
    )
else:
    status_column = next(
        (
            col
            for col in [
                "NoveltyStatus",
                "Status",
            ]
            if col in novelty.columns
        ),
        None,
    )

    if status_column:
        novel_rows = novelty[
            novelty[
                status_column
            ]
            .astype(str)
            .str.contains(
                r"NOVEL|OOD",
                case=False,
                regex=True,
            )
        ]
    else:
        novel_rows = novelty

    if not novel_rows.empty:
        st.dataframe(
            novel_rows,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "OOD candidate ثبت نشده است."
        )


st.warning(
    """
    OOD ≠ Adulteration

    خارج بودن نمونه از دامنه مرجع به‌تنهایی
    تقلب یا ناخالصی را اثبات نمی‌کند.
    """
)


st.divider()


# ============================================================
# SHAP
# ============================================================

st.subheader(
    "🧠 تفسیر مدل و SHAP"
)

shap_path = Path(
    SHAP_PATH
)

if shap_path.exists():
    image_path = (
        BASE_DIR
        / "shap_summary.png"
    )

    if image_path.exists():
        st.image(
            str(image_path),
            caption="SHAP Summary Plot",
            width="stretch",
        )

shap_df = read_csv(
    str(SHAP_PATH)
)

if not shap_df.empty:
    columns = [
        col
        for col in [
            "Rank",
            "Feature",
            "MeanAbsSHAP",
        ]
        if col in shap_df.columns
    ]

    if columns:
        st.markdown(
            "### Top Spectral Features"
        )

        st.dataframe(
            shap_df[columns].head(20),
            hide_index=True,
            use_container_width=True,
        )

st.info(
    """
    SHAP برای تفسیر نقش ویژگی‌ها در تصمیم مدل استفاده می‌شود.
    این نواحی به‌تنهایی biomarker تأییدشده منشأ جغرافیایی نیستند.
    """
)


st.divider()


# ============================================================
# SAMPLE EXPLORER
# ============================================================

st.subheader(
    "🔬 Sample Explorer"
)

if (
    predictions.empty
    or "SampleId"
    not in predictions.columns
):
    st.warning(
        "sample_predictions.csv موجود نیست یا SampleId ندارد."
    )

else:
    sample_ids = (
        pd.to_numeric(
            predictions[
                "SampleId"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    selected_id = st.selectbox(
        "انتخاب نمونه",
        sample_ids,
        key="nmr_origin_sample_explorer",
    )

    selected_df = predictions[
        pd.to_numeric(
            predictions[
                "SampleId"
            ],
            errors="coerce",
        )
        == selected_id
    ]

    if not selected_df.empty:
        row = selected_df.iloc[0]

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:
            st.metric(
                "Sample ID",
                selected_id,
            )

            if "ActualGroup" in row:
                st.write(
                    f"گروه واقعی: "
                    f"**{row['ActualGroup']}**"
                )

        with c2:
            if "PredictedGroup" in row:
                st.write(
                    f"پیش‌بینی مدل: "
                    f"**{row['PredictedGroup']}**"
                )

            if (
                "PredictionConfidence"
                in row
            ):
                st.write(
                    "Confidence: "
                    f"**{float(row['PredictionConfidence']):.4f}**"
                )

        with c3:
            if "Correct" in row:
                correct = normalize_bool(
                    row["Correct"]
                )

                st.write(
                    "پیش‌بینی صحیح: "
                    f"**{'بله' if correct else 'خیر'}**"
                )

        # ----------------------------------------------------
        # OOD
        # ----------------------------------------------------

        sample_ood = (
            pd.DataFrame()
        )

        if (
            not novelty.empty
            and "SampleId"
            in novelty.columns
        ):
            novelty_ids = pd.to_numeric(
                novelty["SampleId"],
                errors="coerce",
            )

            sample_ood = novelty[
                novelty_ids
                == selected_id
            ]

        if not sample_ood.empty:
            st.error(
                "🔍 این نمونه OOD Candidate است."
            )

            st.dataframe(
                sample_ood,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success(
                "این نمونه در دامنه مرجع قرار دارد."
            )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if (
            "PredictionConfidence"
            in row
        ):
            confidence = float(
                row[
                    "PredictionConfidence"
                ]
            )

            st.markdown(
                "### اطمینان پیش‌بینی"
            )

            st.progress(
                min(
                    max(
                        confidence,
                        0.0,
                    ),
                    1.0,
                )
            )

        # ----------------------------------------------------
        # DECISION ENGINE
        # ----------------------------------------------------

        if (
            not decision.empty
            and "SampleId"
            in decision.columns
        ):
            decision_ids = pd.to_numeric(
                decision["SampleId"],
                errors="coerce",
            )

            sample_decision = (
                decision[
                    decision_ids
                    == selected_id
                ]
            )

            if not sample_decision.empty:
                st.markdown(
                    "### Decision Engine"
                )

                st.dataframe(
                    sample_decision,
                    hide_index=True,
                    use_container_width=True,
                )


st.divider()


# ============================================================
# PIPELINE / REPORT / DOCUMENTATION
# ============================================================

with st.expander(
    "📋 Pipeline Run Log",
    expanded=False,
):
    run_output = read_text(
        str(RUN_OUTPUT_PATH)
    )

    if not run_output:
        st.warning(
            "run_output.txt موجود نیست."
        )
    else:
        stages = [
            (
                "Validation",
                "VALIDATION FINISHED",
            ),
            (
                "Preparation",
                "PREPARATION COMPLETED",
            ),
            (
                "Harvest Year",
                "HARVEST YEAR ANALYSIS FINISHED",
            ),
            (
                "PCA",
                "PCA ANALYSIS FINISHED",
            ),
            (
                "Novelty / OOD",
                "MAHALANOBIS NOVELTY / OOD ANALYSIS FINISHED",
            ),
            (
                "Feature Engineering",
                "FEATURE ENGINEERING FINISHED",
            ),
            (
                "Origin Prediction",
                "ORIGIN PREDICTION FINISHED",
            ),
            (
                "Robust Evaluation",
                "ROBUST MODEL EVALUATION FINISHED",
            ),
            (
                "SHAP",
                "SHAP ANALYSIS FINISHED",
            ),
            (
                "Decision Engine",
                "DECISION ENGINE FINISHED",
            ),
            (
                "Final Report",
                "FINAL RESEARCH REPORT FINISHED",
            ),
            (
                "Main Pipeline",
                "MAIN PIPELINE FINISHED",
            ),
        ]

        run_df = pd.DataFrame(
            [
                [
                    name,
                    (
                        "✅"
                        if marker
                        in run_output
                        else "—"
                    ),
                ]
                for name, marker in stages
            ],
            columns=[
                "Stage",
                "Status",
            ],
        )

        st.dataframe(
            run_df,
            hide_index=True,
            use_container_width=True,
        )

        completed = int(
            (
                run_df["Status"]
                == "✅"
            ).sum()
        )

        total = len(run_df)

        st.metric(
            "Pipeline Completion",
            f"{completed} / {total}",
        )

        st.progress(
            completed / total
            if total
            else 0
        )

        with st.expander(
            "نمایش کامل run_output.txt",
            expanded=False,
        ):
            st.code(
                run_output,
                language="text",
            )


with st.expander(
    "📄 Final Research Report",
    expanded=False,
):
    final_report = read_csv(
        str(FINAL_REPORT_PATH)
    )

    if final_report.empty:
        st.warning(
            "final_report.csv موجود نیست یا خالی است."
        )
    else:
        st.dataframe(
            final_report,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            label="⬇️ دانلود Final Report",
            data=final_report.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="final_report.csv",
            mime="text/csv",
        )


with st.expander(
    "📘 README پروژه",
    expanded=False,
):
    readme = read_text(
        str(README_PATH)
    )

    if readme.strip():
        st.markdown(readme)
    else:
        st.warning(
            "README.md موجود نیست."
        )


with st.expander(
    "📝 نتیجه‌گیری علمی",
    expanded=False,
):
    conclusion = read_text(
        str(CONCLUSION_PATH)
    )

    if conclusion.strip():
        st.markdown(conclusion)
    else:
        st.warning(
            "docs/conclusion_fa.md موجود نیست."
        )


st.divider()


# ============================================================
# SCIENTIFIC STATUS
# ============================================================

st.info(
    """
    **وضعیت علمی فعلی**

    مدل منشأ جغرافیایی در وضعیت NOT RELIABLE باقی می‌ماند.
    Sample Explorer، Cross-Year و OOD برای تحلیل و بررسی
    پژوهشی ارائه می‌شوند و نباید به‌عنوان اثبات قطعی منشأ
    یا تقلب تفسیر شوند.
    """
)