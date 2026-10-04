from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "candidate10_predictions.csv"
)

CONFUSION_FILE = (
    ROOT
    / "outputs"
    / "candidate10_confusion_matrix.csv"
)

PAIRS_FILE = (
    ROOT
    / "outputs"
    / "candidate10_confusion_pairs.csv"
)

STABLE_ERRORS_FILE = (
    ROOT
    / "outputs"
    / "candidate10_stable_errors.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    required = {
        "SampleId",
        "ActualClass",
        "PredictedClass",
        "Correct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    classes = sorted(
        set(df["ActualClass"])
        | set(df["PredictedClass"])
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        df["ActualClass"],
        df["PredictedClass"],
        labels=classes,
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    )

    cm_df.index.name = "Actual"

    cm_df.to_csv(
        CONFUSION_FILE
    )

    # ========================================================
    # CONFUSION PAIRS
    # ========================================================

    errors = df[
        df["ActualClass"]
        != df["PredictedClass"]
    ].copy()

    pair_df = (
        errors
        .groupby(
            [
                "ActualClass",
                "PredictedClass",
            ]
        )
        .size()
        .reset_index(
            name="ErrorCount"
        )
    )

    actual_counts = (
        df["ActualClass"]
        .value_counts()
        .to_dict()
    )

    pair_df["ActualCount"] = (
        pair_df["ActualClass"]
        .map(actual_counts)
    )

    pair_df["ErrorRate"] = (
        pair_df["ErrorCount"]
        / pair_df["ActualCount"]
    )

    pair_df = (
        pair_df
        .sort_values(
            "ErrorCount",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    pair_df.to_csv(
        PAIRS_FILE,
        index=False,
    )

    # ========================================================
    # STABLE WRONG SAMPLES
    # ========================================================

    sample_rows = []

    for sample_id, group in (
        df.groupby("SampleId")
    ):

        actual = (
            group["ActualClass"]
            .iloc[0]
        )

        prediction_counts = (
            group["PredictedClass"]
            .value_counts()
        )

        most_frequent = (
            prediction_counts
            .index[0]
        )

        stability = (
            prediction_counts
            .iloc[0]
            / len(group)
        )

        correct_rate = (
            group["Correct"]
            .astype(bool)
            .mean()
        )

        if (
            most_frequent != actual
            and stability >= 0.70
        ):

            sample_rows.append(
                {
                    "SampleId":
                        int(sample_id),

                    "ActualClass":
                        actual,

                    "MostFrequentPrediction":
                        most_frequent,

                    "Stability":
                        float(stability),

                    "CorrectRate":
                        float(correct_rate),

                    "ValidationPredictions":
                        len(group),
                }
            )

    stable_df = (
        pd.DataFrame(
            sample_rows
        )
        .sort_values(
            [
                "Stability",
                "CorrectRate",
            ],
            ascending=[
                False,
                True,
            ],
        )
        if sample_rows
        else pd.DataFrame(
            columns=[
                "SampleId",
                "ActualClass",
                "MostFrequentPrediction",
                "Stability",
                "CorrectRate",
                "ValidationPredictions",
            ]
        )
    )

    stable_df.to_csv(
        STABLE_ERRORS_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("=" * 72)
    print(
        "CANDIDATE 10 ERROR ANALYSIS"
    )
    print("=" * 72)

    print()
    print("Top confusion pairs:")
    print(
        pair_df.head(20).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print(
        "Stable wrong samples (stability >= 70%):"
    )

    if stable_df.empty:
        print("None")
    else:
        print(
            stable_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    print()
    print(
        f"Confusion matrix : {CONFUSION_FILE}"
    )
    print(
        f"Confusion pairs  : {PAIRS_FILE}"
    )
    print(
        f"Stable errors    : {STABLE_ERRORS_FILE}"
    )


if __name__ == "__main__":
    main()