from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CV_N_SPLITS, PLS_COMPONENTS
from .preprocessing import apply_preprocessing


PREPROCESSING_METHODS = [
    "raw",
    "snv",
    "asls_snv",
    "asls_l2",
]

PREPROCESSING_LABELS = {
    "raw": "Raw",
    "snv": "SNV",
    "asls_snv": "AsLS + SNV",
    "asls_l2": "AsLS + L2",
}

MODALITY_NAMES = ["440 nm", "250 nm", "308 nm", "Combined"]


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "plsda",
                PLSRegression(
                    n_components=PLS_COMPONENTS,
                    scale=False,
                    max_iter=500,
                    tol=1e-6,
                ),
            ),
        ]
    )


def _one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return (y[:, None] == classes[None, :]).astype(float)


def _predict_classes(
    model: Pipeline,
    X: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(model.predict(X), dtype=float)
    winner_idx = np.argmax(scores, axis=1)
    pred = classes[winner_idx]

    if scores.shape[1] >= 2:
        ordered = np.sort(scores, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
    else:
        margins = np.abs(scores[:, 0])

    return pred, margins


def evaluate_origin_pipeline(
    X_by_modality: dict[str, np.ndarray],
    y: np.ndarray,
    sample_ids: np.ndarray,
    repeats: int = 20,
    random_state: int = 42,
    preprocessing_methods: list[str] | None = None,
    asls_lambda: float = 1e6,
    asls_p: float = 0.01,
    asls_iterations: int = 10,
    provenance_groups: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessing_methods = preprocessing_methods or PREPROCESSING_METHODS
    y = np.asarray(y, dtype=str)
    sample_ids = np.asarray(sample_ids, dtype=int)
    classes = np.array(sorted(np.unique(y)), dtype=str)
    Y = _one_hot(y, classes)

    processed: dict[str, dict[str, np.ndarray]] = {}
    for method in preprocessing_methods:
        processed[method] = {
            modality: apply_preprocessing(
                X,
                method=method,
                asls_lambda=asls_lambda,
                asls_p=asls_p,
                asls_iterations=asls_iterations,
            )
            for modality, X in X_by_modality.items()
        }
        processed[method]["Combined"] = np.hstack(
            [
                processed[method]["440 nm"],
                processed[method]["250 nm"],
                processed[method]["308 nm"],
            ]
        )

    # Provenance-aware CV: samples that are deterministically linked in a
    # modality must never be split between train and test.  For the current
    # workbook, Samples 39 and 40 are linked at 308 nm (40 = 39 * 0.53),
    # and Combined contains that modality, so they are treated as one CV
    # unit.  Each repeat uses a fresh StratifiedGroupKFold seed.
    if provenance_groups is None:
        provenance_groups = np.asarray(sample_ids, dtype=object).astype(str)
    else:
        provenance_groups = np.asarray(provenance_groups, dtype=object)
        if len(provenance_groups) != len(sample_ids):
            raise ValueError(
                "provenance_groups must have the same length as sample_ids."
            )

    split_list: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    split_counter = 0
    for repeat_id in range(1, repeats + 1):
        cv = StratifiedGroupKFold(
            n_splits=CV_N_SPLITS,
            shuffle=True,
            random_state=random_state + repeat_id - 1,
        )
        for fold_id, (train_idx, test_idx) in enumerate(
            cv.split(
                X_by_modality[MODALITY_NAMES[0]],
                y,
                groups=provenance_groups,
            ),
            start=1,
        ):
            split_counter += 1
            split_list.append((repeat_id, fold_id, train_idx, test_idx))

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for method in preprocessing_methods:
        for modality in MODALITY_NAMES:
            X = processed[method][modality]

            for split_id, (repeat_id, fold_id, train_idx, test_idx) in enumerate(split_list, start=1):
                model = _build_pipeline()
                model.fit(X[train_idx], Y[train_idx])
                pred, margins = _predict_classes(model, X[test_idx], classes)

                metric_rows.append(
                    {
                        "Model": "PLS-DA",
                        "PLSComponents": PLS_COMPONENTS,
                        "Preprocessing": PREPROCESSING_LABELS.get(method, method),
                        "PreprocessingCode": method,
                        "Modality": modality,
                        "Split": split_id,
                        "Repeat": int(repeat_id),
                        "Fold": int(fold_id),
                        "Accuracy": accuracy_score(y[test_idx], pred),
                        # Fold-level balanced accuracy is computed over the
                        # classes that are actually present in this test fold.
                        # This avoids sklearn warnings when the provenance
                        # constraint keeps both G9 samples in one fold.
                        "BalancedAccuracy": recall_score(
                            y[test_idx],
                            pred,
                            labels=np.unique(y[test_idx]),
                            average="macro",
                            zero_division=0,
                        ),
                        "MacroF1": f1_score(
                            y[test_idx],
                            pred,
                            average="macro",
                            zero_division=0,
                        ),
                    }
                )

                for local_pos, idx in enumerate(test_idx):
                    prediction_rows.append(
                        {
                            "Model": "PLS-DA",
                            "PLSComponents": PLS_COMPONENTS,
                            "Preprocessing": PREPROCESSING_LABELS.get(method, method),
                            "PreprocessingCode": method,
                            "Modality": modality,
                            "Split": split_id,
                            "Repeat": int(repeat_id),
                        "Fold": int(fold_id),
                            "SampleId": int(sample_ids[idx]),
                            "ActualGroup": y[idx],
                            "PredictedGroup": pred[local_pos],
                            "Correct": bool(pred[local_pos] == y[idx]),
                            "DecisionMargin": float(margins[local_pos]),
                        }
                    )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_origin_results(split_results: pd.DataFrame) -> pd.DataFrame:
    if split_results.empty:
        return pd.DataFrame()

    summary = (
        split_results.groupby(
            [
                "Model",
                "PLSComponents",
                "Preprocessing",
                "PreprocessingCode",
                "Modality",
            ],
            as_index=False,
        )
        .agg(
            AccuracyMean=("Accuracy", "mean"),
            AccuracyStd=("Accuracy", "std"),
            BalancedAccuracyMean=("BalancedAccuracy", "mean"),
            BalancedAccuracyStd=("BalancedAccuracy", "std"),
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
            TotalSplits=("Split", "count"),
        )
    )

    summary["ChanceBalancedAccuracy"] = 1.0 / 11.0
    return summary.sort_values(
        ["PreprocessingCode", "Modality"]
    ).reset_index(drop=True)


def summarize_repeat_results(cv_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize provenance-aware CV at the repeat level.

    Each sample is predicted exactly once per repeat (across the two folds),
    so this computes Accuracy / Balanced Accuracy / Macro F1 over the full
    out-of-fold set for each repeat. This is preferable to averaging fold BA
    when a provenance group (here G9 samples 39+40) must stay together and
    therefore cannot appear in both folds.
    """
    if cv_predictions.empty:
        return pd.DataFrame()

    metric_rows: list[dict] = []
    group_cols = [
        "Model",
        "PLSComponents",
        "Preprocessing",
        "PreprocessingCode",
        "Modality",
        "Repeat",
    ]

    for keys, frame in cv_predictions.groupby(group_cols, sort=False):
        (
            model_name,
            pls_components,
            preprocessing,
            preprocessing_code,
            modality,
            repeat_id,
        ) = keys

        y_true = frame["ActualGroup"].astype(str).to_numpy()
        y_pred = frame["PredictedGroup"].astype(str).to_numpy()

        metric_rows.append(
            {
                "Model": model_name,
                "PLSComponents": int(pls_components),
                "Preprocessing": preprocessing,
                "PreprocessingCode": preprocessing_code,
                "Modality": modality,
                "Repeat": int(repeat_id),
                "Accuracy": accuracy_score(y_true, y_pred),
                "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
                "MacroF1": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
            }
        )

    repeat_metrics = pd.DataFrame(metric_rows)

    summary = (
        repeat_metrics.groupby(
            [
                "Model",
                "PLSComponents",
                "Preprocessing",
                "PreprocessingCode",
                "Modality",
            ],
            as_index=False,
        )
        .agg(
            AccuracyMean=("Accuracy", "mean"),
            AccuracyStd=("Accuracy", "std"),
            BalancedAccuracyMean=("BalancedAccuracy", "mean"),
            BalancedAccuracyStd=("BalancedAccuracy", "std"),
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
            TotalRepeats=("Repeat", "count"),
        )
    )

    summary["ChanceBalancedAccuracy"] = 1.0 / 11.0
    summary["EvaluationLevel"] = "Repeat-pooled OOF"

    return summary.sort_values(
        ["PreprocessingCode", "Modality"]
    ).reset_index(drop=True)


def fit_descriptive_models(
    X_by_modality: dict[str, np.ndarray],
    y: np.ndarray,
    sample_ids: np.ndarray,
    preprocessing_method: str = "asls_snv",
    asls_lambda: float = 1e6,
    asls_p: float = 0.01,
    asls_iterations: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=str)
    sample_ids = np.asarray(sample_ids, dtype=int)
    classes = np.array(sorted(np.unique(y)), dtype=str)
    Y = _one_hot(y, classes)

    prepared = {
        modality: apply_preprocessing(
            X,
            method=preprocessing_method,
            asls_lambda=asls_lambda,
            asls_p=asls_p,
            asls_iterations=asls_iterations,
        )
        for modality, X in X_by_modality.items()
    }
    prepared["Combined"] = np.hstack(
        [
            prepared["440 nm"],
            prepared["250 nm"],
            prepared["308 nm"],
        ]
    )

    rows: list[dict] = []
    for modality, X in prepared.items():
        model = _build_pipeline()
        model.fit(X, Y)
        pred, margins = _predict_classes(model, X, classes)

        for i in range(len(sample_ids)):
            rows.append(
                {
                    "Model": "PLS-DA",
                    "PLSComponents": PLS_COMPONENTS,
                    "Preprocessing": PREPROCESSING_LABELS.get(
                        preprocessing_method, preprocessing_method
                    ),
                    "PreprocessingCode": preprocessing_method,
                    "Modality": modality,
                    "SampleId": int(sample_ids[i]),
                    "ActualGroup": y[i],
                    "PredictedGroup": pred[i],
                    "Correct": bool(pred[i] == y[i]),
                    "DecisionMargin": float(margins[i]),
                    "FitType": "Full-data descriptive fit",
                    "Warning": "Not a validation prediction; in-sample fit.",
                }
            )

    return pd.DataFrame(rows)
