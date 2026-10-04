import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HPLC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HPLC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = HPLC_DIR / "outputs"
SUMMARY_PATH = OUTPUTS_DIR / "robust_evaluation_summary.csv"

st.set_page_config(page_title="HPLC Origin", page_icon="🌍", layout="wide")

st.title("🌍 منشأ جغرافیایی HPLC")
st.caption("نتایج Provenance-aware برای ارزیابی اکتشافی منشأ جغرافیایی")

if not SUMMARY_PATH.exists():
    st.warning("نتایج robust پیدا نشد. ابتدا از ریشه پروژه `python -m hplc.main` را اجرا کنید.")
    st.stop()

summary = pd.read_csv(SUMMARY_PATH)

st.info(
    "Validation اکنون provenance-aware است: نمونه‌های 39 و 40 به دلیل رابطه 308 nm در یک Fold قرار می‌گیرند. "
    "Balanced Accuracy نهایی از OOF کامل هر Repeat محاسبه شده است."
)

asls = summary[summary["PreprocessingCode"] == "asls_snv"].copy()

if asls.empty:
    st.warning("ردیف‌های AsLS + SNV در خروجی موجود نیستند.")
    st.stop()

view = asls[
    [
        "Modality",
        "AccuracyMean",
        "AccuracyStd",
        "BalancedAccuracyMean",
        "BalancedAccuracyStd",
        "MacroF1Mean",
        "MacroF1Std",
        "ChanceBalancedAccuracy",
    ]
].copy()

view["Accuracy"] = view.apply(lambda r: f"{r.AccuracyMean:.3f} ± {r.AccuracyStd:.3f}", axis=1)
view["Balanced Accuracy"] = view.apply(lambda r: f"{r.BalancedAccuracyMean:.3f} ± {r.BalancedAccuracyStd:.3f}", axis=1)
view["Macro F1"] = view.apply(lambda r: f"{r.MacroF1Mean:.3f} ± {r.MacroF1Std:.3f}", axis=1)

st.dataframe(
    view[["Modality", "Accuracy", "Balanced Accuracy", "Macro F1"]],
    hide_index=True,
    use_container_width=True,
)

fig = px.bar(
    asls,
    x="Modality",
    y="BalancedAccuracyMean",
    error_y="BalancedAccuracyStd",
    title="Balanced Accuracy — AsLS + SNV",
)
fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.warning(
    "این جدول برای مقایسه توصیفی modalityهاست؛ از آن Winner عملیاتی برای سیستم تشخیص منشأ انتخاب نشده است."
)

st.subheader("برداشت از عملکرد")
st.markdown(
    """
    در ارزیابی فعلی، هر modality مقداری اطلاعات جغرافیایی حمل می‌کند، اما تفکیک 11 گروه پایدار و قاطع نیست.
    همپوشانی گروه‌ها، اندازه بسیار کوچک چند کلاس و ساختار سال برداشت باعث می‌شود این نتایج در سطح **اکتشافی** باقی بمانند.
    """
)
