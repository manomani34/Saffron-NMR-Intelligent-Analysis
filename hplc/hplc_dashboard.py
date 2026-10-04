from pathlib import Path
import sys

import pandas as pd
import streamlit as st

HPLC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HPLC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hplc.src.config import HPLC_RAW_PATH
from hplc.src.data_loader import load_hplc_data

OUTPUTS_DIR = HPLC_DIR / "outputs"
PROVENANCE_RELATIONS = OUTPUTS_DIR / "modality_provenance_relations.csv"
FINAL_ASSESSMENT = OUTPUTS_DIR / "hplc_final_assessment.csv"
ROBUST_SUMMARY = OUTPUTS_DIR / "robust_evaluation_summary.csv"

st.set_page_config(
    page_title="HPLC Saffron Analysis",
    page_icon="🧪",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_hplc_data(HPLC_RAW_PATH)


st.title("🧪 تحلیل HPLC زعفران")
st.caption(
    "داشبورد مستقل HPLC — جدا از NMR. "
    "تمرکز: بررسی الگوی جغرافیایی، مقایسه طول موج‌ها و مستندسازی کیفیت و provenance داده."
)

try:
    data = get_data()
except Exception as exc:
    st.error("بارگذاری داده HPLC با خطا مواجه شد.")
    st.exception(exc)
    st.stop()

mapping = data.mapping.copy()
independent = mapping[mapping["Independent"]].copy()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("رکوردهای خام", len(mapping))
with c2:
    st.metric("رکوردهای یکتا پس از حذف تکرار", len(independent))
with c3:
    st.metric("گروه‌های جغرافیایی", mapping["Group"].nunique())
with c4:
    st.metric("نقاط زمانی / طول موج", len(data.wavelengths["440"].time))

st.divider()

st.subheader("وضعیت نهایی دیتاست")
left, right = st.columns([1.15, 1])

with left:
    st.markdown(
        """
        **44 رکورد خام** در Workbook وجود دارد.

        چهار جفت تکراری دقیق عبارت‌اند از **1/2، 24/25، 28/29 و 34/35**؛
        بنابراین در تحلیل آماری **40 مشاهده مستقل** استفاده می‌شود.

        در **308 nm**، نمونه 40 یک نسخه Scale شده از نمونه 39 است:
        **Sample 40 = Sample 39 × 0.53**. این دو در validation در یک گروه provenance
        نگه داشته می‌شوند تا بین train و test شکاف ایجاد نشود.
        """
    )

with right:
    st.warning(
        "Sample 32 در 308 nm یک نقطه نامعتبر دارد. این نقطه در Pipeline با Imputation داخل Fold مدیریت می‌شود."
    )
    st.info(
        "نتایج HPLC فعلی پژوهشی/اکتشافی هستند. با 40 رکورد یکتا پس از حذف تکرار، 11 گروه و چند کلاس 2 نمونه‌ای، "
        "این داده فعلاً مبنای یک classifier عملیاتی و قطعی برای 11 منشأ جغرافیایی نیست."
    )

st.divider()

st.subheader("نتیجه اصلی اعتبارسنجی")
summary = pd.DataFrame()
if ROBUST_SUMMARY.exists():
    summary = pd.read_csv(ROBUST_SUMMARY)

if not summary.empty:
    asls = summary[summary["PreprocessingCode"] == "asls_snv"].copy()
    if not asls.empty:
        show = asls[
            ["Modality", "AccuracyMean", "AccuracyStd", "BalancedAccuracyMean", "BalancedAccuracyStd", "MacroF1Mean", "MacroF1Std"]
        ].copy()
        show = show.rename(
            columns={
                "Modality": "Modality",
                "AccuracyMean": "Accuracy mean",
                "AccuracyStd": "Accuracy std",
                "BalancedAccuracyMean": "BA mean",
                "BalancedAccuracyStd": "BA std",
                "MacroF1Mean": "Macro F1 mean",
                "MacroF1Std": "Macro F1 std",
            }
        )
        st.dataframe(show.round(4), hide_index=True, use_container_width=True)
        st.caption(
            "AsLS + SNV | PLS-DA, 4 components | repeated provenance-aware StratifiedGroupKFold | 2 folds × 20 repeats."
        )

st.markdown(
    "### پیام علمی داشبورد"
)
st.success(
    "داده HPLC برای مقایسه و توصیف الگوهای کروماتوگرافی بین گروه‌ها مفید است، "
    "اما نتایج فعلی برای ادعای یک طبقه‌بندی قطعی 11-گروهی کافی نیستند."
)

st.markdown(
    "از منوی سمت چپ به بخش‌های **Origin، Wavelengths، Sample Explorer، Research Notes** و "
    "**Modality Comparison** بروید."
)
