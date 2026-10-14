from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports" / "color_authenticity_new"

REGISTRY_FILE = REPORT_DIR / "sample_registry.csv"
REVIEW_FILE = REPORT_DIR / "sample_review.csv"
COMPARISON_FILE = REPORT_DIR / "distance_cosine_comparison.csv"


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="اصالت رنگ - نسخه جدید",
    page_icon="🧪",
    layout="wide",
)

st.title("سامانه بررسی اصالت رنگ زعفران")
st.caption("نسخه جدید | مستقل از داشبورد قبلی")


# =========================================================
# Helper functions
# =========================================================

def read_csv_if_exists(file_path):
    """Read a CSV file if it exists."""
    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path, keep_default_na=False)
    except Exception as exc:
        st.error(f"خطا در خواندن فایل {file_path.name}: {exc}")
        return None


def numeric_column(dataframe, column_name):
    """Convert a column to numeric values safely."""
    return pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )


def show_download_button(dataframe, label, file_name):
    """Display a CSV download button."""
    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
    )


def show_table(dataframe, columns, key=None):
    """Display only columns available in the dataframe."""
    available_columns = [
        column for column in columns
        if column in dataframe.columns
    ]

    if not available_columns:
        st.info("ستون قابل نمایشی در این جدول وجود ندارد.")
        return

    st.dataframe(
        dataframe[available_columns],
        use_container_width=True,
        hide_index=True,
        key=key,
    )


# =========================================================
# Load registry and review data
# =========================================================

registry = read_csv_if_exists(REGISTRY_FILE)

if registry is None:
    st.error(
        "فایل فهرست نمونه‌ها پیدا نشد یا قابل خواندن نیست:\n\n"
        f"{REGISTRY_FILE}"
    )
    st.info(
        "ابتدا اسکریپت ساخت فهرست نمونه‌ها را اجرا کن."
    )
    st.stop()

if "initial_group" not in registry.columns:
    st.error(
        "ستون initial_group در فایل sample_registry.csv وجود ندارد."
    )
    st.stop()

review = read_csv_if_exists(REVIEW_FILE)

if review is None:
    review = pd.DataFrame()


# =========================================================
# Summary metrics
# =========================================================

total_samples = len(registry)

saffron_reference_count = int(
    (
        registry["initial_group"]
        == "saffron_reference_candidate"
    ).sum()
)

pure_color_reference_count = int(
    (
        registry["initial_group"]
        == "pure_color_reference_candidate"
    ).sum()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "تعداد کل نمونه‌ها",
        total_samples,
    )

with col2:
    st.metric(
        "نامزدهای مرجع زعفران",
        saffron_reference_count,
    )

with col3:
    st.metric(
        "نامزدهای مرجع رنگ خالص",
        pure_color_reference_count,
    )

st.warning(
    "گروه‌بندی نمونه‌ها مقدماتی است. "
    "تا زمانی که اطلاعات مرجع و منشأ نمونه‌ها تأیید نشده‌اند، "
    "هیچ نمونه‌ای صرفاً بر اساس نتایج این تحلیل، "
    "زعفران اصل یا نمونه تقلبی قطعی اعلام نمی‌شود."
)


# =========================================================
# Tabs
# =========================================================

tab_analysis, tab_samples, tab_review = st.tabs(
    [
        "تحلیل طیفی",
        "فهرست نمونه‌ها",
        "بازبینی و تأیید اطلاعات",
    ]
)


# =========================================================
# Tab 1: Spectral analysis
# =========================================================

with tab_analysis:

    st.subheader("تحلیل فاصله و شباهت طیفی")

    comparison = read_csv_if_exists(COMPARISON_FILE)

    if comparison is None:

        st.error(
            "فایل مقایسه طیفی موجود نیست:\n\n"
            f"{COMPARISON_FILE}"
        )

        st.markdown("برای ساخت گزارش، در پوشه اصلی پروژه اجرا کن:")

        st.code(
            "python -m src.color_authenticity_metric_comparison",
            language="powershell",
        )

    else:

        required_columns = [
            "sample_number",
            "class_value",
            "initial_group",
            "distance_to_saffron_reference",
            "cosine_similarity_to_saffron",
            "metric_agreement",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in comparison.columns
        ]

        if missing_columns:

            st.error(
                "ستون‌های زیر در گزارش مقایسه موجود نیستند:\n\n"
                + ", ".join(missing_columns)
            )

        else:

            # Convert numeric columns
            comparison[
                "distance_to_saffron_reference"
            ] = numeric_column(
                comparison,
                "distance_to_saffron_reference",
            )

            comparison[
                "cosine_similarity_to_saffron"
            ] = numeric_column(
                comparison,
                "cosine_similarity_to_saffron",
            )

            # Translate result labels
            agreement_labels = {
                "both_metrics_close": "هر دو معیار نزدیک",
                "both_metrics_different": "هر دو معیار متفاوت",
                "metrics_disagree_or_borderline": (
                    "اختلاف یا وضعیت مرزی"
                ),
            }

            comparison["نتیجه مقایسه"] = (
                comparison["metric_agreement"]
                .map(agreement_labels)
                .fillna("نامشخص")
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            total_comparison = len(comparison)

            close_count = int(
                (
                    comparison["metric_agreement"]
                    == "both_metrics_close"
                ).sum()
            )

            different_count = int(
                (
                    comparison["metric_agreement"]
                    == "both_metrics_different"
                ).sum()
            )

            borderline_count = int(
                (
                    comparison["metric_agreement"]
                    == "metrics_disagree_or_borderline"
                ).sum()
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "نمونه‌های تحلیل‌شده",
                total_comparison,
            )

            m2.metric(
                "هر دو معیار نزدیک",
                close_count,
            )

            m3.metric(
                "هر دو معیار متفاوت",
                different_count,
            )

            m4.metric(
                "اختلاف یا وضعیت مرزی",
                borderline_count,
            )

            st.divider()

            # -------------------------------------------------
            # Explanation
            # -------------------------------------------------

            st.markdown("### روش تفسیر نتایج")

            st.markdown(
                """
                - **فاصله کمتر:** طیف نمونه بر اساس معیار فاصله به
                  طیف مرجع زعفران نزدیک‌تر است.
                - **شباهت کسینوسی بالاتر:** جهت بردار طیف نمونه
                  به طیف مرجع شباهت بیشتری دارد.
                - **اختلاف معیارها:** دو معیار تصویر یکسانی از
                  نمونه ارائه نمی‌کنند و بررسی بیشتر لازم است.
                """
            )

            st.info(
                "در این تحلیل از مقادیر اکتشافی فاصله 19.996 و "
                "شباهت کسینوسی 0.90 استفاده شده است. "
                "این مقادیر آستانه علمی تأییدشده برای تشخیص اصالت نیستند."
            )

            # -------------------------------------------------
            # Scatter chart
            # -------------------------------------------------

            st.markdown("### نمودار مقایسه نمونه‌ها")

            chart_data = comparison[
                [
                    "initial_group",
                    "distance_to_saffron_reference",
                    "cosine_similarity_to_saffron",
                ]
            ].copy()

            chart_data = chart_data.rename(
                columns={
                    "initial_group": "گروه نمونه",
                    "distance_to_saffron_reference": (
                        "فاصله از مرجع زعفران"
                    ),
                    "cosine_similarity_to_saffron": (
                        "شباهت کسینوسی"
                    ),
                }
            )

            chart_data = chart_data.dropna(
                subset=[
                    "فاصله از مرجع زعفران",
                    "شباهت کسینوسی",
                ]
            )

            if chart_data.empty:

                st.warning(
                    "داده عددی کافی برای رسم نمودار وجود ندارد."
                )

            else:

                st.scatter_chart(
                    chart_data,
                    x="فاصله از مرجع زعفران",
                    y="شباهت کسینوسی",
                    color="گروه نمونه",
                    height=450,
                )

                st.caption(
                    "هر نقطه نشان‌دهنده یک نمونه است. "
                    "رنگ نقاط گروه اولیه نمونه را مشخص می‌کند."
                )

            # -------------------------------------------------
            # Filters
            # -------------------------------------------------

            st.markdown("### جدول نتایج تحلیل")

            available_groups = sorted(
                comparison["initial_group"]
                .astype(str)
                .unique()
                .tolist()
            )

            group_options = ["همه گروه‌ها"] + available_groups

            selected_group = st.selectbox(
                "فیلتر گروه نمونه",
                group_options,
                key="comparison_group_filter",
            )

            result_options = [
                "همه نتایج",
                "هر دو معیار نزدیک",
                "هر دو معیار متفاوت",
                "اختلاف یا وضعیت مرزی",
                "نامشخص",
            ]

            selected_result = st.selectbox(
                "فیلتر نتیجه مقایسه",
                result_options,
                key="comparison_result_filter",
            )

            filtered_comparison = comparison.copy()

            if selected_group != "همه گروه‌ها":

                filtered_comparison = filtered_comparison[
                    filtered_comparison["initial_group"].astype(str)
                    == selected_group
                ]

            if selected_result != "همه نتایج":

                filtered_comparison = filtered_comparison[
                    filtered_comparison["نتیجه مقایسه"]
                    == selected_result
                ]

            filtered_comparison = filtered_comparison.sort_values(
                by="distance_to_saffron_reference",
                na_position="last",
            )

            comparison_columns = [
                "sample_number",
                "class_value",
                "title",
                "initial_group",
                "distance_to_saffron_reference",
                "cosine_similarity_to_saffron",
                "نتیجه مقایسه",
                "verification_status",
            ]

            show_table(
                filtered_comparison,
                comparison_columns,
            )

            st.caption(
                f"تعداد نتایج پس از فیلتر: "
                f"{len(filtered_comparison)}"
            )

            show_download_button(
                filtered_comparison,
                "دانلود نتایج فیلترشده CSV",
                "color_authenticity_metric_comparison.csv",
            )

            # -------------------------------------------------
            # Borderline cases
            # -------------------------------------------------

            st.divider()

            st.markdown(
                "### نمونه‌های نیازمند بررسی بیشتر"
            )

            borderline = comparison[
                comparison["metric_agreement"]
                == "metrics_disagree_or_borderline"
            ].copy()

            if borderline.empty:

                st.success(
                    "در گزارش فعلی نمونه‌ای در گروه اختلاف "
                    "یا وضعیت مرزی وجود ندارد."
                )

            else:

                borderline = borderline.sort_values(
                    by="distance_to_saffron_reference",
                    na_position="last",
                )

                show_table(
                    borderline,
                    [
                        "sample_number",
                        "class_value",
                        "title",
                        "initial_group",
                        "distance_to_saffron_reference",
                        "cosine_similarity_to_saffron",
                        "نتیجه مقایسه",
                    ],
                )

                show_download_button(
                    borderline,
                    "دانلود نمونه‌های نیازمند بررسی",
                    "color_authenticity_borderline_samples.csv",
                )


# =========================================================
# Tab 2: Sample registry
# =========================================================

with tab_samples:

    st.subheader("فهرست کامل نمونه‌ها")

    registry_groups = sorted(
        registry["initial_group"]
        .astype(str)
        .unique()
        .tolist()
    )

    registry_group_options = (
        ["همه گروه‌ها"] + registry_groups
    )

    selected_registry_group = st.selectbox(
        "فیلتر گروه نمونه",
        registry_group_options,
        key="registry_group_filter",
    )

    filtered_registry = registry.copy()

    if selected_registry_group != "همه گروه‌ها":

        filtered_registry = filtered_registry[
            filtered_registry["initial_group"].astype(str)
            == selected_registry_group
        ]

    registry_columns = [
        "source_row",
        "sample_number",
        "title",
        "description",
        "class_value",
        "initial_group",
        "verification_status",
    ]

    show_table(
        filtered_registry,
        registry_columns,
    )

    st.caption(
        f"تعداد نمونه‌های نمایش‌داده‌شده: "
        f"{len(filtered_registry)}"
    )

    show_download_button(
        filtered_registry,
        "دانلود فهرست نمونه‌ها CSV",
        "color_authenticity_sample_registry.csv",
    )


# =========================================================
# Tab 3: Review and verification
# =========================================================

with tab_review:

    st.subheader("بازبینی و تأیید اطلاعات نمونه‌ها")

    if review.empty:

        st.info(
            "فایل بازبینی نمونه‌ها موجود نیست یا هنوز آماده نشده است."
        )

    else:

        st.markdown(
            """
            این جدول برای ثبت و پیگیری بررسی کارشناسی نمونه‌هاست.
            مقادیر اولیه به‌تنهایی به معنی تأیید اصالت نیستند.
            """
        )

        review_columns = [
            "sample_number",
            "class_value",
            "review_decision",
            "confirmed_authenticity",
            "confirmed_color_type",
            "confirmed_for_reference",
            "review_notes",
        ]

        show_table(
            review,
            review_columns,
        )

        show_download_button(
            review,
            "دانلود فایل بازبینی CSV",
            "color_authenticity_sample_review.csv",
        )

        st.warning(
            "این صفحه فعلاً اطلاعات بازبینی را نمایش می‌دهد و "
            "تغییرات جدول را مستقیماً در فایل ذخیره نمی‌کند. "
            "برای جلوگیری از تغییر ناخواسته داده‌های مرجع، "
            "تأیید کارشناسی باید در فایل بازبینی ثبت شود."
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "گزارش‌های نسخه جدید در پوشه "
    "reports/color_authenticity_new نگهداری می‌شوند. "
    "این صفحه مستقل از داشبورد و مدل‌های قدیمی است."
)
