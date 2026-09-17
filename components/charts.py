import pandas as pd
import streamlit as st


def show_bar(
    df,
    index_column,
    value_column,
    title=None,
):

    if df is None or df.empty:
        st.info("داده‌ای برای نمودار موجود نیست.")
        return

    chart = df[
        [index_column, value_column]
    ].copy()

    chart[index_column] = chart[
        index_column
    ].astype(str)

    if title:
        st.markdown(
            f"### {title}"
        )

    st.bar_chart(
        chart.set_index(
            index_column
        )[value_column]
    )


def show_group_distribution(
    group_df,
):

    if group_df.empty:
        return

    show_bar(
        group_df,
        "Group",
        "Samples",
        "توزیع نمونه‌ها بین مناطق",
    )


def show_year_distribution(
    year_df,
):

    if year_df.empty:
        return

    show_bar(
        year_df,
        "HarvestYear",
        "Samples",
        "توزیع نمونه‌ها بین سال‌های برداشت",
    )
