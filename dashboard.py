import streamlit as st

from components.common import (
    apply_page_style,
    get_dataset_info,
    get_origin_metrics,
    get_ood_count,
    file_time,
    show_refresh_button,
    DATASET_PATH,
    ROBUST_SUMMARY_PATH,
    NOVELTY_PATH,
    COLOR_DASHBOARD_JSON,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Saffron NMR Research Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_page_style()

dataset = get_dataset_info()
origin = get_origin_metrics()
ood_count = get_ood_count()
color_available = COLOR_DASHBOARD_JSON.exists()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🧪 Saffron NMR")
    st.caption("سامانه پژوهشی تحلیل زعفران مبتنی بر NMR")

    st.divider()

    st.markdown("### صفحات اصلی")
    st.caption("از منوی Pages در Sidebar انتخاب کنید:")
    st.write("🌍 منشأ جغرافیایی")
    st.write("📅 سال برداشت")
    st.write("🎨 اصالت و رنگ مصنوعی")
    st.write("🔬 پژوهش و جزئیات فنی")

    st.divider()

    st.markdown("### وضعیت فایل‌ها")
    st.caption(f"Dataset: {file_time(DATASET_PATH)}")
    st.caption(f"Robust: {file_time(ROBUST_SUMMARY_PATH)}")
    st.caption(f"OOD: {file_time(NOVELTY_PATH)}")
    st.caption(f"Authenticity: {file_time(COLOR_DASHBOARD_JSON)}")

    st.divider()
    show_refresh_button()


# ============================================================
# HEADER
# ============================================================

st.title("🧪 سامانه پژوهشی تحلیل زعفران مبتنی بر NMR")
st.caption("Intelligent NMR-Based Saffron Analysis")

st.markdown(
    """
**🌍 منشأ جغرافیایی**

**📅 سال برداشت**

**🎨 اصالت و شواهد رنگ مصنوعی**
"""
)

st.divider()


# ============================================================
# GLOBAL KPIs
# ============================================================

st.subheader("نمای کلی Dataset")

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("رکوردهای اولیه", dataset["records"])

with k2:
    st.metric("نمونه‌های مستقل", dataset["independent"])

with k3:
    st.metric("مناطق جغرافیایی", dataset["groups"])

with k4:
    st.metric("نقاط طیفی", f"{dataset['spectral_points']:,}")

with k5:
    st.metric("نمونه‌های OOD", ood_count)

st.divider()


# ============================================================
# THREE RESEARCH DOMAINS
# ============================================================

st.subheader("سه محور اصلی پژوهش")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("## 🌍 منشأ جغرافیایی")
    st.write(
        """
        بررسی قابلیت طیف NMR برای تفکیک منشأ جغرافیایی
        نمونه‌های زعفران.
        """
    )

    st.markdown("### خلاصه عملکرد مدل")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", origin.get("accuracy", "—"))
    with m2:
        st.metric("Balanced Accuracy", origin.get("ba", "—"))
    with m3:
        st.metric("Macro F1", origin.get("f1", "—"))

    st.caption(
        f"TopK: **{origin.get('top_k', '—')}** | "
        f"Accuracy SD: **{origin.get('accuracy_std', '—')}** | "
        f"BA SD: **{origin.get('ba_std', '—')}** | "
        f"F1 SD: **{origin.get('f1_std', '—')}**"
    )

    st.warning(
        "مدل فعلی برای استفاده عملیاتی قابل اتکا نیست."
    )

with c2:
    st.markdown("## 📅 سال برداشت")
    st.write(
        """
        بررسی اثر سال برداشت و پایداری الگوهای طیفی
        بین سال‌ها.
        """
    )

    years = dataset["years"]
    year_text = " / ".join(str(year) for year in years) if years else "—"
    st.metric("سال‌های موجود", year_text)

    region_year = dataset["region_year"]
    if not region_year.empty:
        common_regions = int(
            (((region_year > 0).sum(axis=1)) >= 2).sum()
        )
    else:
        common_regions = 0

    st.metric("مناطق دارای هر دو سال", common_regions)
    st.info(
        "جزئیات کامل Year-CV، Within-Region و Region×Year "
        "در صفحه سال برداشت قرار دارد."
    )

with c3:
    st.markdown("## 🎨 اصالت و رنگ مصنوعی")
    st.write(
        """
        بررسی شواهد طیفی مرتبط با رنگ مصنوعی در
        Pilot Dataset.
        """
    )

    st.metric(
        "وضعیت تحلیل",
        "Available" if color_available else "Not Available",
    )

    st.warning(
        "این بخش Pilot / Exploratory است و هنوز برای "
        "استفاده عملیاتی مستقل اعتبارسنجی نشده است."
    )


st.divider()


# ============================================================
# RESEARCH STATUS
# ============================================================

st.subheader("وضعیت فعلی پژوهش")

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown("### 🌍 منشأ جغرافیایی")
    st.error("NOT RELIABLE")
    st.caption(
        "داده مرجع بیشتر و اعتبارسنجی بین‌سالانه مورد نیاز است."
    )

with s2:
    st.markdown("### 📅 سال برداشت")
    st.info("UNDER ANALYSIS")
    st.caption(
        "اثر سال برداشت و Cross-Year در حال بررسی است."
    )

with s3:
    st.markdown("### 🎨 اصالت / رنگ مصنوعی")
    if color_available:
        st.warning("PILOT / EXPLORATORY")
    else:
        st.info("NOT AVAILABLE")
    st.caption(
        "اعتبارسنجی مستقل هنوز انجام نشده است."
    )


st.divider()


# ============================================================
# DATA QUALITY
# ============================================================

st.subheader("وضعیت کیفیت داده")

q1, q2, q3, q4 = st.columns(4)

with q1:
    if dataset["missing"] == 0:
        st.success("Missing Values = 0")
    else:
        st.warning(f"Missing Values = {dataset['missing']}")

with q2:
    if dataset["invalid"] == 0:
        st.success("Spectral Values = Valid")
    else:
        st.warning(f"Invalid Values = {dataset['invalid']}")

with q3:
    st.metric(
        "نقاط طیفی",
        f"{dataset['spectral_points']:,}",
    )

with q4:
    groups = dataset["group_counts"]
    if groups.empty:
        distribution = "—"
    else:
        distribution = (
            f"{int(groups['Samples'].min())} تا "
            f"{int(groups['Samples'].max())}"
        )
    st.metric("نمونه در هر منطقه", distribution)


st.divider()


# ============================================================
# DISTRIBUTIONS
# ============================================================

left, right = st.columns(2)

with left:
    st.subheader("توزیع مناطق")
    groups = dataset["group_counts"]

    if groups.empty:
        st.info("اطلاعات مناطق موجود نیست.")
    else:
        st.bar_chart(groups.set_index("Group")["Samples"])
        st.dataframe(
            groups,
            hide_index=True,
            use_container_width=True,
        )

with right:
    st.subheader("توزیع سال برداشت")
    year_counts = dataset["year_counts"].copy()

    if year_counts.empty:
        st.info("اطلاعات سال برداشت موجود نیست.")
    else:
        year_chart = year_counts[["HarvestYear", "Samples"]].copy()
        year_chart["HarvestYear"] = (
            year_chart["HarvestYear"].astype(int).astype(str)
        )
        st.bar_chart(year_chart.set_index("HarvestYear")["Samples"])
        st.dataframe(
            year_counts,
            hide_index=True,
            use_container_width=True,
        )


st.divider()


# ============================================================
# REGION × YEAR
# ============================================================

st.subheader("منطقه × سال برداشت")

region_year = dataset["region_year"]

if region_year.empty:
    st.info("جدول Region × Harvest Year موجود نیست.")
else:
    region_year_display = (
        region_year.copy().reset_index()
    )

    st.dataframe(
        region_year_display,
        hide_index=True,
        use_container_width=True,
    )

    year_presence = (region_year > 0).sum(axis=1)
    common_regions = int((year_presence >= 2).sum())

    st.info(
        f"مناطق دارای نمونه در حداقل دو سال: "
        f"**{common_regions} / {dataset['groups']}**"
    )


st.divider()


st.info(
    """
این صفحه نمای کلی سامانه است.
جزئیات کامل مدل‌های منشأ، Cross-Year/OOD، SHAP،
Sample Explorer، تحلیل سال برداشت، Region×Year و
اصالت در صفحات تخصصی پروژه ارائه می‌شوند.
"""
)

st.caption("Saffron NMR Research Dashboard")
