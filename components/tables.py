import pandas as pd
import streamlit as st


def show_dataframe(
    df,
    title=None,
    max_rows=None,
):

    if df is None or df.empty:
        st.info("داده‌ای برای نمایش موجود نیست.")
        return

    if title:
        st.markdown(
            f"### {title}"
        )

    display = df.copy()

    if max_rows is not None:
        display = display.head(
            max_rows
        )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
    )


def confusion_matrix(
    predictions,
):

    required = {
        "ActualGroup",
        "PredictedGroup",
    }

    if predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        predictions.columns
    ):
        return pd.DataFrame()

    actual = predictions[
        "ActualGroup"
    ].astype(str)

    predicted = predictions[
        "PredictedGroup"
    ].astype(str)

    labels = sorted(
        set(actual) | set(predicted)
    )

    matrix = pd.crosstab(
        actual,
        predicted,
        rownames=["Actual"],
        colnames=["Predicted"],
    )

    return matrix.reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )


def error_summary(
    predictions,
):

    required = {
        "ActualGroup",
        "PredictedGroup",
    }

    if predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        predictions.columns
    ):
        return pd.DataFrame()

    df = predictions.copy()

    df["CorrectCalc"] = (
        df["ActualGroup"].astype(str)
        == df["PredictedGroup"].astype(str)
    )

    summary = (
        df.groupby("ActualGroup")
        .agg(
            Samples=("ActualGroup", "size"),
            Correct=("CorrectCalc", "sum"),
        )
        .reset_index()
    )

    summary["Errors"] = (
        summary["Samples"]
        - summary["Correct"]
    )

    summary["Accuracy"] = (
        summary["Correct"]
        / summary["Samples"]
    )

    return summary.sort_values(
        "Accuracy"
    )
