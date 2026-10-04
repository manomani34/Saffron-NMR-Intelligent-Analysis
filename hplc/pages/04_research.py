import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HPLC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HPLC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hplc.src.config import HPLC_RAW_PATH, MAPPING_PATH
from hplc.src.data_loader import load_hplc_data

OUTPUTS_DIR = HPLC_DIR / "outputs"
RELATIONS = OUTPUTS_DIR / "modality_provenance_relations.csv"
FORMULAS = OUTPUTS_DIR / "modality_provenance_formula_samples.csv"
INTEGRITY = OUTPUTS_DIR / "modality_provenance_integrity.csv"
SUMMARY = OUTPUTS_DIR / "robust_evaluation_summary.csv"

st.set_page_config(page_title="HPLC Research Notes", page_icon="📚", layout="wide")

st.title("📚 مستندات و وضعیت علمی HPLC")

data = load_hplc_data(HPLC_RAW_PATH)
mapping = data.mapping.copy()
independent = mapping[mapping["Independent"]].copy()

st.subheader("منابع")
st.write(f"Raw workbook: `{HPLC_RAW_PATH}`")
st.write(f"Authoritative mapping: `{MAPPING_PATH}`")

st.subheader("ساختار نهایی داده")
structure = pd.DataFrame(
    [
        ["رکورد خام", len(mapping)],
        ["رکوردهای یکتا پس از حذف تکرار", len(independent)],
        ["گروه جغرافیایی", mapping["Group"].nunique()],
        ["طول موج‌ها", "440 / 250 / 308 nm"],
        ["نقاط زمانی", len(data.wavelengths["440"].time)],
        ["بازه زمانی (min)", f"{data.wavelengths['440'].time[0]:.8f} – {data.wavelengths['440'].time[-1]:.2f}"],
    ],
    columns=["Item", "Value"],
)
st.dataframe(structure, hide_index=True, use_container_width=True)

st.subheader("Provenance")
if RELATIONS.exists():
    rel = pd.read_csv(RELATIONS)
    st.dataframe(rel, hide_index=True, use_container_width=True)

st.subheader("Formula-bearing samples")
if FORMULAS.exists():
    st.dataframe(pd.read_csv(FORMULAS), hide_index=True, use_container_width=True)

st.subheader("Integrity flags")
if INTEGRITY.exists():
    st.dataframe(pd.read_csv(INTEGRITY), hide_index=True, use_container_width=True)

st.subheader("تفسیر provenance")
st.markdown(
    """
    **در هر سه طول موج:** چهار جفت تکراری دقیق وجود دارد: 1/2، 24/25، 28/29 و 34/35.

    **فقط در 308 nm:** Sample 40 یک نسخه Scale شده از Sample 39 با ضریب 0.53 است.
    بنابراین این دو برای validation 308 و Combined نباید بین train و test جدا شوند.

    **Sample 32 / 308 nm:** تنها یک نقطه نامعتبر دارد و حذف کل رکورد لازم نیست؛
    مدیریت این نقطه در Pipeline انجام می‌شود.
    """
)

st.subheader("دامنه علمی نتیجه")
st.warning(
    "در این Pilot، HPLC به‌عنوان ابزار characterization و مقایسه الگوهای جغرافیایی تفسیر می‌شود، "
    "نه به‌عنوان classifier عملیاتی نهایی برای 11 منشأ."
)

st.info(
    "سال برداشت هدف اصلی مدل نیست، اما ساختار آن در داده بسیار قوی است و می‌تواند بر تفسیر جغرافیا اثر بگذارد. "
    "بنابراین نتیجه جغرافیایی باید همراه با این محدودیت گزارش شود."
)

if SUMMARY.exists():
    summary = pd.read_csv(SUMMARY)
    st.subheader("متریک‌های provenance-aware")
    st.dataframe(
        summary[summary["PreprocessingCode"] == "asls_snv"][
            ["Modality", "AccuracyMean", "BalancedAccuracyMean", "MacroF1Mean", "TotalRepeats"]
        ].round(4),
        hide_index=True,
        use_container_width=True,
    )
