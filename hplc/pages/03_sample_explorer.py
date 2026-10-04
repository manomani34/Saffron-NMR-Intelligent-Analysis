from pathlib import Path
import pandas as pd
import streamlit as st

HPLC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HPLC_DIR.parent
OUTPUTS_DIR = HPLC_DIR / "outputs"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from hplc.src.config import HPLC_RAW_PATH
from hplc.src.data_loader import load_hplc_data

st.set_page_config(
    page_title="HPLC Sample Explorer",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_hplc_data(HPLC_RAW_PATH)


def read_csv(filename: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


data = get_data()
mapping = data.mapping
sample_ids = mapping["SampleId"].astype(int).tolist()

st.title("🔬 Sample Explorer HPLC")

selected_id = st.selectbox("Sample ID", sample_ids)
meta = mapping[mapping["SampleId"] == selected_id].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Sample ID", int(meta["SampleId"]))
with c2:
    st.metric("Group", meta["Group"])
with c3:
    st.metric("Harvest Year", int(meta["HarvestYear"]))
with c4:
    st.metric("HPLC Number", str(meta["HPLCNumber"]))

st.write(
    f"Region: **{meta['Region']}** | Sample Name: **{meta['SampleName']}** | "
    f"Independent measurement: **{'Yes' if bool(meta['Independent']) else 'No'}**"
)

if not bool(meta["Independent"]):
    st.warning(
        "این SampleId طبق Mapping یک ردیف تکراری از همان اندازه‌گیری HPLC است و در ارزیابی آماری مستقل استفاده نمی‌شود."
    )

cv_summary = read_csv("sample_cv_summary.csv")
if not cv_summary.empty:
    st.subheader("نتیجه Cross-Validation برای این نمونه")
    sample_cv = cv_summary[cv_summary["SampleId"] == selected_id].copy()
    if sample_cv.empty:
        st.info("برای این نمونه نتیجه CV موجود نیست.")
    else:
        st.dataframe(
            sample_cv[
                [
                    "Preprocessing",
                    "Modality",
                    "ActualGroup",
                    "MostFrequentPredictedGroup",
                    "CorrectRate",
                    "MeanDecisionMargin",
                    "PredictionCount",
                ]
            ].style.format(
                {
                    "CorrectRate": "{:.3f}",
                    "MeanDecisionMargin": "{:.4f}",
                    "PredictionCount": "{:.0f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "CorrectRate حاصل پیش‌بینی‌های OOF در 40 split ارزیابی Robust است؛ این بخش validation-based است و با fit روی کل داده تفاوت دارد."
    )

for code, title in [
    ("440", "440 nm"),
    ("250", "250 nm"),
    ("308", "308 nm"),
]:
    item = data.wavelengths[code]
    idx = item.sample_ids.tolist().index(selected_id)

    df = pd.DataFrame(
        {
            "Time (min)": item.time,
            title: item.values[idx],
        }
    ).set_index("Time (min)")

    st.markdown(f"### {title}")
    st.line_chart(
        df,
        use_container_width=True,
        height=250,
    )

st.info(
    "نمودارهای بالا داده خام HPLC را نشان می‌دهند. جدول Cross-Validation از خروجی main.py خوانده می‌شود و در زمان باز شدن صفحه، مدل دوباره اجرا نمی‌شود."
)
