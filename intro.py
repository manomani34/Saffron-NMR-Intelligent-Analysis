from __future__ import annotations

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="سامانه هوشمند تحلیل زعفران",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# INTRO PAGE
# ============================================================

intro_page = st.Page(
    "intro.py",
    title="صفحه اصلی",
    icon="🏠",
    url_path="",
    default=True,
)


# ============================================================
# NMR PAGES
# ============================================================

nmr_home = st.Page(
    "dashboard.py",
    title="داشبورد NMR",
    icon="🧪",
    url_path="nmr",
    visibility="hidden",
)

nmr_origin = st.Page(
    "pages/01_origin.py",
    title="منشأ جغرافیایی",
    icon="🌍",
    url_path="nmr-origin",
    visibility="hidden",
)

nmr_year = st.Page(
    "pages/02_harvest_year.py",
    title="سال برداشت",
    icon="📅",
    url_path="nmr-harvest-year",
    visibility="hidden",
)

nmr_authenticity = st.Page(
    "pages/03_authenticity.py",
    title="اصالت و رنگ مصنوعی",
    icon="🎨",
    url_path="nmr-authenticity",
    visibility="hidden",
)

nmr_research = st.Page(
    "pages/04_research.py",
    title="پژوهش و جزئیات فنی",
    icon="🔬",
    url_path="nmr-research",
    visibility="hidden",
)


# ============================================================
# HPLC PAGES
# ============================================================

hplc_home = st.Page(
    "hplc/hplc_dashboard.py",
    title="نمای کلی HPLC",
    icon="📊",
    url_path="hplc",
    visibility="hidden",
)

hplc_origin = st.Page(
    "hplc/pages/01_origin.py",
    title="منشأ جغرافیایی",
    icon="🌍",
    url_path="hplc-origin",
    visibility="hidden",
)

hplc_wavelengths = st.Page(
    "hplc/pages/02_wavelengths.py",
    title="طول موج‌ها و متابولیت‌ها",
    icon="📈",
    url_path="hplc-wavelengths",
    visibility="hidden",
)

hplc_sample_explorer = st.Page(
    "hplc/pages/03_sample_explorer.py",
    title="کاوش نمونه‌ها",
    icon="🔬",
    url_path="hplc-sample-explorer",
    visibility="hidden",
)

hplc_research = st.Page(
    "hplc/pages/04_research.py",
    title="وضعیت علمی و پژوهشی",
    icon="📚",
    url_path="hplc-research",
    visibility="hidden",
)

hplc_modality = st.Page(
    "hplc/pages/05_modality_comparison.py",
    title="مقایسه طول موج‌ها",
    icon="⚖️",
    url_path="hplc-modality-comparison",
    visibility="hidden",
)


# ============================================================
# INTRO CONTENT
# ============================================================

def render_intro():

    st.markdown(
        """
        <style>

        /* ====================================================
           Hide Sidebar
        ==================================================== */

        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="collapsedControl"] {
            display: none !important;
        }

        #MainMenu {
            visibility: hidden !important;
        }

        footer {
            visibility: hidden !important;
        }


        /* ====================================================
           RTL
        ==================================================== */

        html,
        body,
        .stApp {
            direction: rtl !important;
        }


        /* ====================================================
           Main Container
        ==================================================== */

        .block-container {
            max-width: 1100px !important;
            padding-top: 5rem !important;
            padding-bottom: 4rem !important;
        }


        /* ====================================================
           Font
        ==================================================== */

        body,
        p,
        span,
        label,
        button {
            font-family: Tahoma, Arial, sans-serif !important;
        }


        /* ====================================================
           Header
        ==================================================== */

        .intro-header {
            text-align: center;
            margin-bottom: 55px;
        }

        .intro-icon {
            font-size: 64px;
            margin-bottom: 18px;
        }

        .intro-title {
            font-size: 38px;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 12px;
        }

        .intro-subtitle {
            font-size: 18px;
            color: #6b7280;
            line-height: 2;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        """
        <div class="intro-header">

            <div class="intro-icon">🧪</div>

            <div class="intro-title">
                سامانه هوشمند تحلیل زعفران
            </div>

            <div class="intro-subtitle">
                سامانه پژوهشی تحلیل داده‌های زعفران
                بر پایه روش‌های NMR و HPLC
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # CARDS
    # ========================================================

    _, nmr_col, hplc_col, _ = st.columns(
        [0.7, 3, 3, 0.7],
        gap="large",
    )


    # ========================================================
    # NMR CARD
    # ========================================================

    with nmr_col:

        with st.container(border=True):

            st.markdown("## 🧪 تحلیل NMR")

            st.write(
                """
                تحلیل منشأ جغرافیایی،
                سال برداشت،
                Novelty / OOD
                و بررسی اصالت نمونه‌ها
                بر پایه داده‌های NMR
                """
            )

            if st.button(
                "ورود به تحلیل NMR",
                key="open_nmr",
                use_container_width=True,
            ):
                st.switch_page(nmr_home)


    # ========================================================
    # HPLC CARD
    # ========================================================

    with hplc_col:

        with st.container(border=True):

            st.markdown("## 📈 تحلیل HPLC")

            st.write(
                """
                تحلیل کروماتوگرام‌های زعفران
                در طول موج‌های 250، 308 و 440 نانومتر
                و بررسی پروفایل جغرافیایی
                """
            )

            if st.button(
                "ورود به تحلیل HPLC",
                key="open_hplc",
                use_container_width=True,
            ):
                st.switch_page(hplc_home)


# ============================================================
# NAVIGATION
# ============================================================

pg = st.navigation(
    [
        # ----------------------------------------------------
        # Intro
        # ----------------------------------------------------

        intro_page,

        # ----------------------------------------------------
        # NMR
        # ----------------------------------------------------

        nmr_home,
        nmr_origin,
        nmr_year,
        nmr_authenticity,
        nmr_research,

        # ----------------------------------------------------
        # HPLC
        # ----------------------------------------------------

        hplc_home,
        hplc_origin,
        hplc_wavelengths,
        hplc_sample_explorer,
        hplc_research,
        hplc_modality,
    ],
    position="hidden",
)


# ============================================================
# RUN
# ============================================================

if pg == intro_page:
    render_intro()
else:
    pg.run()