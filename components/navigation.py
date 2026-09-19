from __future__ import annotations

import streamlit as st


# ============================================================
# NMR NAVIGATION
# ============================================================

def render_nmr_navigation() -> None:

    with st.sidebar:

        st.header("🧪 Saffron NMR")

        st.caption(
            "سامانه پژوهشی تحلیل زعفران مبتنی بر NMR"
        )

        st.divider()

        st.page_link(
            "intro.py",
            label="صفحه اصلی",
            icon="🏠",
        )

        st.divider()

        st.markdown("### صفحات NMR")

        st.page_link(
            "dashboard.py",
            label="داشبورد NMR",
            icon="🧪",
        )

        st.page_link(
            "pages/01_origin.py",
            label="منشأ جغرافیایی",
            icon="🌍",
        )

        st.page_link(
            "pages/02_harvest_year.py",
            label="سال برداشت",
            icon="📅",
        )

        st.page_link(
            "pages/03_authenticity.py",
            label="اصالت و رنگ مصنوعی",
            icon="🎨",
        )

        st.page_link(
            "pages/04_research.py",
            label="پژوهش و جزئیات فنی",
            icon="🔬",
        )


# ============================================================
# HPLC NAVIGATION
# ============================================================

def render_hplc_navigation() -> None:

    with st.sidebar:

        st.header("📈 HPLC")

        st.caption(
            "سامانه تحلیل HPLC زعفران"
        )

        st.divider()

        st.page_link(
            "intro.py",
            label="صفحه اصلی",
            icon="🏠",
        )

        st.divider()

        st.markdown("### صفحات HPLC")

        st.page_link(
            "hplc/hplc_dashboard.py",
            label="نمای کلی HPLC",
            icon="📊",
        )

        st.page_link(
            "hplc/pages/01_origin.py",
            label="منشأ جغرافیایی",
            icon="🌍",
        )

        st.page_link(
            "hplc/pages/02_wavelengths.py",
            label="طول موج‌ها و متابولیت‌ها",
            icon="📈",
        )

        st.page_link(
            "hplc/pages/03_sample_explorer.py",
            label="کاوش نمونه‌ها",
            icon="🔬",
        )

        st.page_link(
            "hplc/pages/04_research.py",
            label="وضعیت علمی و پژوهشی",
            icon="📚",
        )

        st.page_link(
            "hplc/pages/05_modality_comparison.py",
            label="مقایسه طول موج‌ها",
            icon="⚖️",
        )