from __future__ import annotations

import pandas as pd


def build_scenario_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a readable scenario identifier:
        Preprocessing | Modality
    """
    result = df.copy()

    result["Scenario"] = (
        result["Preprocessing"].astype(str)
        + " | "
        + result["Modality"].astype(str)
    )

    return result


def build_sample_robustness(
    cv_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build sample-level robustness summary from repeated CV predictions.

    Each independent sample is evaluated once per repeat in a
    2-fold repeated stratified CV design. With 20 repeats,
    each sample has 20 validation predictions per scenario.
    """

    required = {
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "SampleId",
        "ActualGroup",
        "PredictedGroup",
        "Correct",
        "DecisionMargin",
    }

    if cv_predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        cv_predictions.columns
    ):
        return pd.DataFrame()

    df = build_scenario_column(
        cv_predictions
    )

    rows: list[dict] = []

    group_cols = [
        "Scenario",
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "SampleId",
        "ActualGroup",
    ]

    for keys, group in df.groupby(
        group_cols,
        sort=True,
    ):
        (
            scenario,
            preprocessing,
            preprocessing_code,
            modality,
            sample_id,
            actual_group,
        ) = keys

        prediction_counts = (
            group["PredictedGroup"]
            .astype(str)
            .value_counts()
        )

        if prediction_counts.empty:
            continue

        most_frequent_prediction = (
            prediction_counts.index[0]
        )

        dominant_count = int(
            prediction_counts.iloc[0]
        )

        prediction_count = int(
            len(group)
        )

        stability = (
            dominant_count / prediction_count
            if prediction_count
            else 0.0
        )

        correct_values = pd.to_numeric(
            group["Correct"],
            errors="coerce",
        )

        margin_values = pd.to_numeric(
            group["DecisionMargin"],
            errors="coerce",
        )

        correct_rate = (
            correct_values.mean()
            if not correct_values.empty
            else 0.0
        )

        mean_margin = (
            margin_values.mean()
            if not margin_values.empty
            else None
        )

        median_margin = (
            margin_values.median()
            if not margin_values.empty
            else None
        )

        majority_correct = (
            str(most_frequent_prediction)
            == str(actual_group)
        )

        if majority_correct:
            if correct_rate >= 0.75:
                robustness_status = (
                    "STABLE CORRECT"
                )
            else:
                robustness_status = (
                    "MOSTLY CORRECT / MIXED"
                )
        else:
            if stability >= 0.75:
                robustness_status = (
                    "STABLE INCORRECT"
                )
            else:
                robustness_status = (
                    "MIXED / UNSTABLE"
                )

        rows.append(
            {
                "Scenario": scenario,
                "Preprocessing": preprocessing,
                "PreprocessingCode": (
                    preprocessing_code
                ),
                "Modality": modality,
                "SampleId": int(sample_id),
                "ActualGroup": str(
                    actual_group
                ),
                "MostFrequentPredictedGroup": str(
                    most_frequent_prediction
                ),
                "PredictionCount": prediction_count,
                "CorrectRate": float(
                    correct_rate
                )
                if pd.notna(correct_rate)
                else 0.0,
                "Stability": float(
                    stability
                ),
                "MeanDecisionMargin": (
                    float(mean_margin)
                    if pd.notna(mean_margin)
                    else None
                ),
                "MedianDecisionMargin": (
                    float(median_margin)
                    if pd.notna(median_margin)
                    else None
                ),
                "MajorityPredictionCorrect": (
                    bool(majority_correct)
                ),
                "RobustnessStatus": (
                    robustness_status
                ),
            }
        )

    return pd.DataFrame(rows)


def build_class_error_analysis(
    sample_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize sample-level robustness by actual class.
    """

    required = {
        "Scenario",
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "SampleId",
        "ActualGroup",
        "CorrectRate",
        "MajorityPredictionCorrect",
        "Stability",
        "MeanDecisionMargin",
    }

    if sample_analysis.empty:
        return pd.DataFrame()

    if not required.issubset(
        sample_analysis.columns
    ):
        return pd.DataFrame()

    rows: list[dict] = []

    group_cols = [
        "Scenario",
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "ActualGroup",
    ]

    for keys, group in sample_analysis.groupby(
        group_cols,
        sort=True,
    ):
        (
            scenario,
            preprocessing,
            preprocessing_code,
            modality,
            actual_group,
        ) = keys

        samples = int(
            group["SampleId"].nunique()
        )

        majority_correct_count = int(
            group[
                "MajorityPredictionCorrect"
            ].astype(bool).sum()
        )

        mostly_correct_count = int(
            (
                pd.to_numeric(
                    group["CorrectRate"],
                    errors="coerce",
                )
                >= 0.75
            ).sum()
        )

        rows.append(
            {
                "Scenario": scenario,
                "Preprocessing": preprocessing,
                "PreprocessingCode": (
                    preprocessing_code
                ),
                "Modality": modality,
                "ActualGroup": str(
                    actual_group
                ),
                "Samples": samples,
                "MajorityCorrectSamples": (
                    majority_correct_count
                ),
                "MajorityCorrectRate": (
                    majority_correct_count / samples
                    if samples
                    else 0.0
                ),
                "MostlyCorrectSamples_75pct": (
                    mostly_correct_count
                ),
                "MeanSampleCorrectRate": float(
                    pd.to_numeric(
                        group["CorrectRate"],
                        errors="coerce",
                    ).mean()
                ),
                "MeanStability": float(
                    pd.to_numeric(
                        group["Stability"],
                        errors="coerce",
                    ).mean()
                ),
                "MeanDecisionMargin": float(
                    pd.to_numeric(
                        group[
                            "MeanDecisionMargin"
                        ],
                        errors="coerce",
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_confusion_pairs(
    cv_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate repeated-CV confusion events.

    ConfusionCount:
        Number of validation prediction events.

    ConfusionRate:
        ConfusionCount divided by all validation predictions
        in the selected scenario.

    WithinActualRate:
        ConfusionCount divided by all validation predictions
        belonging to the same ActualGroup.

    UniqueSamplesAffected:
        Number of independent samples that experienced this
        actual -> predicted pair at least once.
    """

    required = {
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "SampleId",
        "ActualGroup",
        "PredictedGroup",
    }

    if cv_predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        cv_predictions.columns
    ):
        return pd.DataFrame()

    df = build_scenario_column(
        cv_predictions
    )

    rows: list[dict] = []

    # Total validation events per scenario
    scenario_totals = (
        df.groupby(
            "Scenario"
        )
        .size()
        .to_dict()
    )

    group_cols = [
        "Scenario",
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "ActualGroup",
        "PredictedGroup",
    ]

    grouped = (
        df.groupby(
            group_cols,
            sort=True,
        )
        .size()
        .reset_index(
            name="ConfusionCount"
        )
    )

    for _, group in grouped.iterrows():

        scenario = str(
            group["Scenario"]
        )

        preprocessing = str(
            group["Preprocessing"]
        )

        preprocessing_code = str(
            group["PreprocessingCode"]
        )

        modality = str(
            group["Modality"]
        )

        actual_group = str(
            group["ActualGroup"]
        )

        predicted_group = str(
            group["PredictedGroup"]
        )

        confusion_count = int(
            group["ConfusionCount"]
        )

        total_predictions = int(
            scenario_totals.get(
                scenario,
                0,
            )
        )

        # Total validation events for this ActualGroup
        actual_group_total = int(
            df[
                (
                    df["Scenario"]
                    == scenario
                )
                & (
                    df["ActualGroup"].astype(str)
                    == actual_group
                )
            ].shape[0]
        )

        # Number of independent samples affected
        unique_samples = int(
            df[
                (
                    df["Scenario"]
                    == scenario
                )
                & (
                    df["ActualGroup"].astype(str)
                    == actual_group
                )
                & (
                    df["PredictedGroup"].astype(str)
                    == predicted_group
                )
            ]["SampleId"].nunique()
        )

        rows.append(
            {
                "Scenario": scenario,
                "Preprocessing": preprocessing,
                "PreprocessingCode": (
                    preprocessing_code
                ),
                "Modality": modality,
                "ActualGroup": actual_group,
                "PredictedGroup": predicted_group,
                "ConfusionCount": confusion_count,
                "ConfusionRate": (
                    confusion_count
                    / total_predictions
                    if total_predictions
                    else 0.0
                ),
                "WithinActualRate": (
                    confusion_count
                    / actual_group_total
                    if actual_group_total
                    else 0.0
                ),
                "UniqueSamplesAffected": (
                    unique_samples
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["IsError"] = (
        result["ActualGroup"].astype(str)
        != result["PredictedGroup"].astype(str)
    )

    return result.sort_values(
        [
            "Scenario",
            "IsError",
            "ConfusionCount",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_confusion_matrix_long(
    cv_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one long-format row per
    Scenario × Actual × Predicted.
    """

    pairs = build_confusion_pairs(
        cv_predictions
    )

    if pairs.empty:
        return pd.DataFrame()

    return pairs[
        [
            "Scenario",
            "Preprocessing",
            "PreprocessingCode",
            "Modality",
            "ActualGroup",
            "PredictedGroup",
            "ConfusionCount",
            "ConfusionRate",
            "UniqueSamplesAffected",
        ]
    ].copy()


def build_class_prediction_summary(
    cv_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Class-level repeated-CV performance directly
    from validation events.
    """

    required = {
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "ActualGroup",
        "Correct",
        "DecisionMargin",
    }

    if cv_predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        cv_predictions.columns
    ):
        return pd.DataFrame()

    df = build_scenario_column(
        cv_predictions
    )

    result = (
        df.groupby(
            [
                "Scenario",
                "Preprocessing",
                "PreprocessingCode",
                "Modality",
                "ActualGroup",
            ],
            as_index=False,
        )
        .agg(
            ValidationPredictions=(
                "Correct",
                "size",
            ),
            CorrectPredictions=(
                "Correct",
                "sum",
            ),
            MeanCorrect=(
                "Correct",
                "mean",
            ),
            MeanDecisionMargin=(
                "DecisionMargin",
                "mean",
            ),
        )
    )

    result["ErrorPredictions"] = (
        result["ValidationPredictions"]
        - result["CorrectPredictions"]
    )

    result["ErrorRate"] = (
        1.0
        - result["MeanCorrect"]
    )

    return result.sort_values(
        [
            "Scenario",
            "ErrorRate",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_normalized_confusion_matrix(
    cv_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build row-normalized confusion information.

    WithinActualRate is the fraction of predictions for each
    ActualGroup that were assigned to each PredictedGroup.
    """

    required = {
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "ActualGroup",
        "PredictedGroup",
    }

    if cv_predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        cv_predictions.columns
    ):
        return pd.DataFrame()

    df = build_scenario_column(
        cv_predictions
    )

    grouped = (
        df.groupby(
            [
                "Scenario",
                "Preprocessing",
                "PreprocessingCode",
                "Modality",
                "ActualGroup",
                "PredictedGroup",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "ConfusionCount"
            }
        )
    )

    actual_totals = (
        grouped.groupby(
            [
                "Scenario",
                "ActualGroup",
            ],
            as_index=False,
        )[
            "ConfusionCount"
        ]
        .sum()
        .rename(
            columns={
                "ConfusionCount":
                    "ActualGroupTotal"
            }
        )
    )

    result = grouped.merge(
        actual_totals,
        on=[
            "Scenario",
            "ActualGroup",
        ],
        how="left",
    )

    result["WithinActualRate"] = (
        result["ConfusionCount"]
        / result["ActualGroupTotal"]
    )

    result["IsCorrect"] = (
        result["ActualGroup"].astype(str)
        == result["PredictedGroup"].astype(str)
    )

    return result[
        [
            "Scenario",
            "Preprocessing",
            "PreprocessingCode",
            "Modality",
            "ActualGroup",
            "PredictedGroup",
            "ConfusionCount",
            "ActualGroupTotal",
            "WithinActualRate",
            "IsCorrect",
        ]
    ].sort_values(
        [
            "Scenario",
            "ActualGroup",
            "WithinActualRate",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_group_performance_summary(
    cv_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-group repeated-CV performance.

    Recall:
        Correct validation predictions /
        all validation predictions for that group.

    UniqueSamples:
        Number of independent samples represented by the group.
    """

    required = {
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "SampleId",
        "ActualGroup",
        "PredictedGroup",
        "Correct",
        "DecisionMargin",
    }

    if cv_predictions.empty:
        return pd.DataFrame()

    if not required.issubset(
        cv_predictions.columns
    ):
        return pd.DataFrame()

    df = build_scenario_column(
        cv_predictions
    )

    df["CorrectNumeric"] = pd.to_numeric(
        df["Correct"],
        errors="coerce",
    )

    df["DecisionMarginNumeric"] = pd.to_numeric(
        df["DecisionMargin"],
        errors="coerce",
    )

    result = (
        df.groupby(
            [
                "Scenario",
                "Preprocessing",
                "PreprocessingCode",
                "Modality",
                "ActualGroup",
            ],
            as_index=False,
        )
        .agg(
            ValidationPredictions=(
                "ActualGroup",
                "size",
            ),
            CorrectPredictions=(
                "CorrectNumeric",
                "sum",
            ),
            UniqueSamples=(
                "SampleId",
                "nunique",
            ),
            MeanDecisionMargin=(
                "DecisionMarginNumeric",
                "mean",
            ),
        )
    )

    result["ErrorPredictions"] = (
        result["ValidationPredictions"]
        - result["CorrectPredictions"]
    )

    result["Recall"] = (
        result["CorrectPredictions"]
        / result["ValidationPredictions"]
    )

    result["ErrorRate"] = (
        1.0
        - result["Recall"]
    )

    result["MeanDecisionMargin"] = (
        result["MeanDecisionMargin"]
        .fillna(0.0)
    )

    return result.sort_values(
        [
            "Scenario",
            "Recall",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )