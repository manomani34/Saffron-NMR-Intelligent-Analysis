import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

HPLC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HPLC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hplc.src.config import (
    CROCIN_WINDOW,
    HPLC_RAW_PATH,
    PICROCROCCIN_REFERENCE,
    SAFRANAL_REFERENCE,
)
from hplc.src.data_loader import load_hplc_data


st.set_page_config(
    page_title="HPLC Wavelengths",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_hplc_data(HPLC_RAW_PATH)


st.title("📈 طول موج‌ها و متابولیت‌ها")

data = get_data()
mapping = data.mapping

st.info(
    "تحلیل آماری اصلی از کل بازه 0.016667 تا 30 دقیقه استفاده می‌کند. "
    "پنجره‌ها و زمان‌های متابولیتی در این صفحه فقط برای تفسیر و بررسی توصیفی نمایش داده می‌شوند."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Crocin — 440 nm", "14 تا 30 دقیقه")

with c2:
    st.metric("Picrocrocin — 250 nm", "≈ 14.3 دقیقه")

with c3:
    st.metric("Safranal — 308 nm", "≈ 30.9 دقیقه")

st.warning(
    "30.9 دقیقه خارج از محدوده فعلی فایل است؛ آخرین نقطه داده 30.0 دقیقه است. "
    "این مورد قبل از هر تحلیل اختصاصی سافرانال باید از نظر روش/محور زمان تأیید شود."
)

st.divider()

sample_options = mapping["SampleId"].astype(int).tolist()
selected_id = st.selectbox("نمونه", sample_options, index=0)

meta = mapping[mapping["SampleId"] == selected_id].iloc[0]

st.write(
    f"**Sample {selected_id}** — Group: **{meta['Group']}** | "
    f"Region: **{meta['Region']}** | Sample: **{meta['SampleName']}** | "
    f"HPLC: **{meta['HPLCNumber']}**"
)

for code, title in [
    ("440", "Crocin — 440 nm"),
    ("250", "Picrocrocin — 250 nm"),
    ("308", "Safranal — 308 nm"),
]:
    item = data.wavelengths[code]
    row_idx = item.sample_ids.tolist().index(selected_id)

    chart_df = pd.DataFrame(
        {
            "Time (min)": item.time,
            "Absorbance": item.values[row_idx],
        }
    ).set_index("Time (min)")

    st.markdown(f"### {title}")
    st.line_chart(
        chart_df,
        use_container_width=True,
        height=260,
    )

st.subheader("شاخص‌های توصیفی")

row440 = data.wavelengths["440"].values[
    data.wavelengths["440"].sample_ids.tolist().index(selected_id)
]
row250 = data.wavelengths["250"].values[
    data.wavelengths["250"].sample_ids.tolist().index(selected_id)
]
row308 = data.wavelengths["308"].values[
    data.wavelengths["308"].sample_ids.tolist().index(selected_id)
]

idx440 = (data.wavelengths["440"].time >= CROCIN_WINDOW[0]) & (
    data.wavelengths["440"].time <= CROCIN_WINDOW[1]
)
idx250 = int(
    np.argmin(
        np.abs(
            data.wavelengths["250"].time
            - PICROCROCCIN_REFERENCE
        )
    )
)

d1, d2, d3 = st.columns(3)

with d1:
    st.metric(
        "Max absorbance در Crocin window",
        f"{np.nanmax(row440[idx440]):.6g}",
    )

with d2:
    st.metric(
        f"Absorbance نزدیک {data.wavelengths['250'].time[idx250]:.4f} min",
        f"{row250[idx250]:.6g}",
    )

with d3:
    st.metric(
        "Max absorbance کل Safranal 308",
        f"{np.nanmax(row308):.6g}",
    )
