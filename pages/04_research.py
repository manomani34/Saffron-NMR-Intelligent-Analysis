import streamlit as st

from components.common import (
    apply_page_style,
    read_csv,
    read_text,
    file_time,
    ROBUST_RESULTS_PATH,
    NOVELTY_PATH,
    SHAP_PATH,
    FINAL_REPORT_PATH,
    RUN_OUTPUT_PATH,
    README_PATH,
    CONCLUSION_PATH,
    PCA_SCORES_PLOT,
    PCA_SCREE_PLOT,
    NOVELTY_PLOT,
    SHAP_PLOT,
)


st.set_page_config(
    page_title="پژوهش و فنی",
    page_icon="🔬",
    layout="wide",
)

apply_page_style()


st.title(
    "🔬 پژوهش و جزئیات فنی"
)

st.caption(
    "Scientific & Technical Details"
)

st.divider()


# ============================================================
# PCA
# ============================================================

st.subheader(
    "📈 PCA"
)

c1, c2 = st.columns(2)

with c1:

    if PCA_SCORES_PLOT.exists():

        st.image(
            str(PCA_SCORES_PLOT),
            caption="PCA Scores",
            use_container_width=True,
        )

    else:

        st.info(
            "PCA Scores موجود نیست."
        )


with c2:

    if PCA_SCREE_PLOT.exists():

        st.image(
            str(PCA_SCREE_PLOT),
            caption="PCA Scree Plot",
            use_container_width=True,
        )

    else:

        st.info(
            "PCA Scree Plot موجود نیست."
        )


st.divider()


# ============================================================
# OOD
# ============================================================

st.subheader(
    "🔍 Novelty / OOD"
)

novelty = read_csv(
    NOVELTY_PATH
)

if novelty.empty:

    st.info(
        "نتایج OOD موجود نیست."
    )

else:

    st.dataframe(
        novelty,
        hide_index=True,
        use_container_width=True,
    )

    if NOVELTY_PLOT.exists():

        st.image(
            str(NOVELTY_PLOT),
            caption="Mahalanobis Novelty Detection",
            use_container_width=True,
        )


st.divider()


# ============================================================
# SHAP
# ============================================================

st.subheader(
    "🧠 SHAP"
)

c1, c2 = st.columns(2)

with c1:

    if SHAP_PLOT.exists():

        st.image(
            str(SHAP_PLOT),
            caption="SHAP Summary",
            use_container_width=True,
        )

    else:

        st.info(
            "SHAP plot موجود نیست."
        )


with c2:

    shap = read_csv(
        SHAP_PATH
    )

    if shap.empty:

        st.info(
            "SHAP results موجود نیست."
        )

    else:

        st.dataframe(
            shap.head(20),
            hide_index=True,
            use_container_width=True,
        )


st.divider()


# ============================================================
# ROBUST RAW RESULTS
# ============================================================

st.subheader(
    "Robust Evaluation — Detailed"
)

robust = read_csv(
    ROBUST_RESULTS_PATH
)

if robust.empty:

    st.info(
        "robust_evaluation_results.csv موجود نیست."
    )

else:

    st.dataframe(
        robust,
        hide_index=True,
        use_container_width=True,
    )


st.divider()


# ============================================================
# FINAL REPORT
# ============================================================

st.subheader(
    "📄 Final Research Report"
)

report = read_csv(
    FINAL_REPORT_PATH
)

if report.empty:

    st.info(
        "Final report موجود نیست."
    )

else:

    st.dataframe(
        report,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ دانلود Final Report",
        data=report.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="final_report.csv",
        mime="text/csv",
    )


st.divider()


# ============================================================
# RUN LOG
# ============================================================

st.subheader(
    "📋 Run Log"
)

run_output = read_text(
    RUN_OUTPUT_PATH
)

st.caption(
    f"آخرین اجرا: {file_time(RUN_OUTPUT_PATH)}"
)

if not run_output:

    st.info(
        "run_output.txt موجود نیست."
    )

else:

    stages = [
        "VALIDATION FINISHED",
        "PREPARATION COMPLETED",
        "HARVEST YEAR ANALYSIS FINISHED",
        "PCA ANALYSIS FINISHED",
        "MAHALANOBIS NOVELTY / OOD ANALYSIS FINISHED",
        "FEATURE ENGINEERING FINISHED",
        "ORIGIN PREDICTION FINISHED",
        "ROBUST MODEL EVALUATION FINISHED",
        "SHAP ANALYSIS FINISHED",
        "DECISION ENGINE FINISHED",
        "FINAL RESEARCH REPORT FINISHED",
        "MAIN PIPELINE FINISHED",
    ]

    completed = sum(
        marker in run_output
        for marker in stages
    )

    st.metric(
        "Pipeline Completion",
        f"{completed} / {len(stages)}",
    )

    st.progress(
        completed / len(stages)
    )

    with st.expander(
        "نمایش کامل run_output.txt"
    ):

        st.code(
            run_output,
            language="text",
        )


st.divider()


# ============================================================
# DOCUMENTATION
# ============================================================

st.subheader(
    "📘 Documentation"
)

readme = read_text(
    README_PATH
)

if readme:

    with st.expander(
        "README.md"
    ):

        st.markdown(
            readme
        )


conclusion = read_text(
    CONCLUSION_PATH
)

if conclusion:

    with st.expander(
        "نتیجه‌گیری علمی"
    ):

        st.markdown(
            conclusion
        )


st.info(
    """
    این صفحه برای جزئیات فنی و پژوهشی است.
    کاربر عادی داشبورد لازم نیست برای استفاده از سامانه
    وارد این بخش شود.
    """
)
