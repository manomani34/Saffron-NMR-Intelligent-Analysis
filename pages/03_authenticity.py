from pathlib import Path
import json
import pandas as pd
import streamlit as st

from components.common import apply_page_style


BASE_DIR = Path(__file__).resolve().parents[1]

COLOR_DASHBOARD_JSON = (
    BASE_DIR
    / "reports"
    / "color_dashboard"
    / "color_dashboard_result.json"
)

COLOR_DASHBOARD_CSV = (
    BASE_DIR
    / "reports"
    / "color_dashboard"
    / "color_dashboard_samples.csv"
)

apply_page_style()


@st.cache_data(show_spinner=False)
def load_result():
    if not COLOR_DASHBOARD_JSON.exists():
        return {}

    try:
        with open(
            COLOR_DASHBOARD_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


st.title("🎨 اصالت و شواهد رنگ مصنوعی")

st.caption(
    "Pilot analysis for artificial-color evidence in saffron spectra"
)

st.divider()


result = load_result()

if not result:
    st.warning(
        "نتایج تحلیل color.csv هنوز تولید نشده‌اند."
    )
    st.code(
        "python -m src.color_dashboard_result",
        language="powershell",
    )
    st.stop()


analysis = result.get("analysis", {}) or {}
threshold = result.get("threshold", {}) or {}
summary = result.get("summary", {}) or {}
samples = result.get("samples", []) or {}
region = analysis.get("region", {}) or {}


st.warning(
    """
    این بخش Pilot / Exploratory است.
    شاخص‌های آن از همان Pilot Dataset به‌دست آمده‌اند و
    برای استفاده عملیاتی یا ادعای تشخیص قطعی تقلب،
    به اعتبارسنجی مستقل نیاز دارند.
    """
)


# ============================================================
# OVERVIEW
# ============================================================

st.subheader("نمای کلی تحلیل")

total_samples = summary.get(
    "totalSamples",
    len(samples) if isinstance(samples, list) else 0,
)

status_counts = summary.get(
    "statusCounts",
    {},
) or {}

positive_count = status_counts.get(
    "positive",
    0,
)

suspicious_count = status_counts.get(
    "suspicious",
    0,
)

reference_count = status_counts.get(
    "reference",
    0,
)

threshold_value = threshold.get(
    "value",
    None,
)

balanced_accuracy = threshold.get(
    "balancedAccuracy",
    analysis.get("balancedAccuracy"),
)

sensitivity = threshold.get(
    "sensitivity",
    analysis.get("sensitivity"),
)

specificity = threshold.get(
    "specificity",
    analysis.get("specificity"),
)


st.caption(
    f"ناحیه مورد استفاده: "
    f"{region.get('lowerPpm', 5.0)} "
    f"تا "
    f"{region.get('upperPpm', 9.0)} ppm"
)


c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "کل نمونه‌ها",
        total_samples,
    )

with c2:
    st.metric(
        "شناسایی رنگ مصنوعی",
        positive_count,
    )

with c3:
    st.metric(
        "نمونه‌های مشکوک",
        suspicious_count,
    )

with c4:
    st.metric(
        "نمونه‌های مرجع",
        reference_count,
    )

with c5:
    st.metric(
        "آستانه اکتشافی",
        fmt(threshold_value),
    )


st.divider()


# ============================================================
# EXPLORATORY VALIDATION
# ============================================================

st.subheader("📊 اعتبارسنجی اکتشافی")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Balanced Accuracy",
        fmt(balanced_accuracy),
    )

with c2:
    st.metric(
        "Sensitivity",
        fmt(sensitivity),
    )

with c3:
    st.metric(
        "Specificity",
        fmt(specificity),
    )

st.caption(
    "این شاخص‌ها اکتشافی‌اند و اعتبارسنجی مستقل محسوب نمی‌شوند."
)


st.divider()


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

st.subheader("توزیع تصمیم‌ها")

decision_counts = summary.get(
    "decisionCounts",
    {},
) or {}

if decision_counts:
    decision_df = pd.DataFrame(
        [
            {
                "Decision": key,
                "Samples": value,
            }
            for key, value
            in decision_counts.items()
        ]
    )

    st.dataframe(
        decision_df,
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        decision_df.set_index(
            "Decision"
        )["Samples"]
    )


st.divider()


# ============================================================
# SAMPLE RESULTS
# ============================================================

st.subheader("نتایج نمونه‌ها")

sample_df = pd.DataFrame(samples)

if sample_df.empty:
    st.info(
        "نتیجه‌ای برای نمونه‌ها موجود نیست."
    )
else:
    display_columns = [
        "number",
        "name",
        "class",
        "decision",
        "status",
        "score",
        "confidence",
        "artificialColorPercent",
    ]

    available = [
        c
        for c in display_columns
        if c in sample_df.columns
    ]

    display_df = sample_df[available].copy()

    rename = {
        "number": "شماره",
        "name": "نمونه",
        "class": "کلاس مرجع",
        "decision": "تصمیم",
        "status": "وضعیت",
        "score": "امتیاز",
        "confidence": "اعتماد",
        "artificialColorPercent":
            "درصد رنگ مصنوعی",
    }

    display_df = display_df.rename(
        columns=rename
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )


st.divider()


# ============================================================
# SAMPLE INSPECTOR
# ============================================================

st.subheader("🔬 بررسی یک نمونه")

if sample_df.empty:
    st.info(
        "داده‌ای برای بررسی نمونه وجود ندارد."
    )
else:
    numbers = (
        sample_df["number"]
        .dropna()
        .tolist()
    )

    selected_number = st.selectbox(
        "نمونه را انتخاب کنید",
        numbers,
        key="nmr_authenticity_sample_selector",
        format_func=lambda value: (
            f"#{int(value)} — "
            f"{sample_df.loc[
                sample_df['number'] == value,
                'name'
            ].iloc[0]}"
        )
        if "name" in sample_df.columns
        else f"#{int(value)}",
    )

    selected_rows = sample_df[
        sample_df["number"] == selected_number
    ]

    if not selected_rows.empty:
        selected = selected_rows.iloc[0]

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Sample",
                int(selected_number),
            )

        with b:
            st.metric(
                "امتیاز رنگ",
                fmt(selected.get("score")),
            )

        with c:
            st.metric(
                "Confidence",
                str(
                    selected.get(
                        "confidence",
                        "—",
                    )
                ),
            )

        st.markdown(
            f"""
**نام نمونه:** {selected.get('name', '—')}

**کلاس مرجع:** {selected.get('class', '—')}

**تصمیم:** {selected.get('decision', '—')}

**وضعیت:** {selected.get('status', '—')}

**دلیل:** {selected.get('reason', '—')}
"""
        )

        artificial_percent = selected.get(
            "artificialColorPercent",
            None,
        )

        if pd.notna(artificial_percent):
            st.write(
                "درصد رنگ مصنوعی ثبت‌شده در Metadata: "
                f"**{float(artificial_percent):.1f}%**"
            )


st.divider()


# ============================================================
# CSV / LIMITATIONS
# ============================================================

if COLOR_DASHBOARD_CSV.exists():
    st.download_button(
        label="⬇️ دانلود نتایج تشخیص رنگ",
        data=COLOR_DASHBOARD_CSV.read_bytes(),
        file_name="color_dashboard_samples.csv",
        mime="text/csv",
    )

st.markdown("### محدودیت‌های علمی")

for limitation in result.get(
    "limitations",
    [],
) or []:
    st.warning(limitation)


st.info(
    """
    **نکته علمی**

    خارج بودن نمونه از الگوی مرجع، به‌تنهایی اثبات
    تقلب یا ناخالصی نیست. برای ادعای Adulteration Detection
    باید Ground Truth معتبر و اعتبارسنجی مستقل فراهم شود.
    """
)
