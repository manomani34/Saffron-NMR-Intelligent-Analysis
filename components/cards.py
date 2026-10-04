import streamlit as st


def metric_card(
    label,
    value,
    help_text=None,
):

    st.metric(
        label=label,
        value=value,
        help=help_text,
    )


def section_status(
    title,
    status,
    description=None,
):

    st.markdown(f"### {title}")

    if status == "ok":
        st.success(
            description
            or "وضعیت تأیید شده"
        )

    elif status == "warning":
        st.warning(
            description
            or "نیازمند بررسی"
        )

    elif status == "error":
        st.error(
            description
            or "نتیجه تأیید نشده"
        )

    else:
        st.info(
            description
            or "اطلاعات در دسترس است"
        )


def three_status_cards(
    first,
    second,
    third,
):

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            first[0],
            first[1],
        )

    with c2:
        metric_card(
            second[0],
            second[1],
        )

    with c3:
        metric_card(
            third[0],
            third[1],
        )
