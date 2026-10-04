from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HPLC_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = HPLC_DIR / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "robust_evaluation_summary.csv"

st.set_page_config(page_title="HPLC Modality Comparison", page_icon="🧭", layout="wide")

st.title("🧭 مقایسه Modalityها")
st.caption("مقایسه توصیفی 250، 308، 440 و تحلیل Combined با validation provenance-aware")

if not SUMMARY_PATH.exists():
    st.warning("فایل `robust_evaluation_summary.csv` پیدا نشد.")
    st.stop()

summary = pd.read_csv(SUMMARY_PATH)

prep = st.selectbox(
    "پیش‌پردازش",
    sorted(summary["Preprocessing"].dropna().unique().tolist()),
    index=(sorted(summary["Preprocessing"].dropna().unique().tolist()).index("AsLS + SNV") if "AsLS + SNV" in summary["Preprocessing"].values else 0),
)

view = summary[summary["Preprocessing"] == prep].copy()

st.dataframe(
    view[
        [
            "Modality",
            "AccuracyMean",
            "AccuracyStd",
            "BalancedAccuracyMean",
            "BalancedAccuracyStd",
            "MacroF1Mean",
            "MacroF1Std",
        ]
    ].round(4),
    hide_index=True,
    use_container_width=True,
)

metric = st.radio(
    "متریک",
    ["AccuracyMean", "BalancedAccuracyMean", "MacroF1Mean"],
    format_func=lambda x: {
        "AccuracyMean": "Accuracy",
        "BalancedAccuracyMean": "Balanced Accuracy",
        "MacroF1Mean": "Macro F1",
    }[x],
    horizontal=True,
)

fig = px.bar(
    view,
    x="Modality",
    y=metric,
    error_y=metric.replace("Mean", "Std"),
    title=f"{prep} — {metric}",
)
fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.warning(
    "این صفحه رتبه‌بندی عملیاتی ارائه نمی‌کند. هدف فقط نشان دادن تفاوت توصیفی modalityها تحت یک validation مشترک است."
)

st.info(
    "ترکیب سه طول موج (Combined) الزاماً به بهبود Balanced Accuracy منجر نشده است؛ بنابراین Accuracy بالاتر آن به‌تنهایی به معنی عملکرد جغرافیایی بهتر نیست."
)
