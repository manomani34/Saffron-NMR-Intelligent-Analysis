from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

X_PATH = Path(
    "data/processed/color_X.npy"
)

PPM_PATH = Path(
    "data/processed/color_ppm.npy"
)

META_PATH = Path(
    "data/processed/color_preprocessed.csv"
)

OUTPUT_DIR = Path(
    "reports/color_knn_diagnostic"
)

COLOR_REGION = (
    5.0,
    9.0,
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    X = np.load(
        X_PATH
    )

    ppm = np.load(
        PPM_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    return X, ppm, meta


# ============================================================
# DUPLICATE GROUPS
# ============================================================

def build_duplicate_groups(X):

    groups = np.empty(
        X.shape[0],
        dtype=int,
    )

    signatures = {}

    next_group = 0

    for i, spectrum in enumerate(X):

        signature = spectrum.tobytes()

        if signature not in signatures:
            signatures[signature] = next_group
            next_group += 1

        groups[i] = signatures[signature]

    return groups


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR DATASET KNN DIAGNOSTIC")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X, ppm, meta = load_data()

    print()
    print(
        f"Original X : {X.shape}"
    )

    # --------------------------------------------------------
    # 5–9 PPM
    # --------------------------------------------------------

    region_mask = (
        (ppm >= COLOR_REGION[0])
        & (ppm <= COLOR_REGION[1])
    )

    X_region = X[
        :,
        region_mask,
    ]

    ppm_region = ppm[
        region_mask
    ]

    print(
        f"Color region : "
        f"{COLOR_REGION[0]} - "
        f"{COLOR_REGION[1]} ppm"
    )

    print(
        f"Region X : "
        f"{X_region.shape}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = (
        meta[
            "artificial_color"
        ]
        .astype(int)
        .to_numpy()
    )

    print()
    print("[1] TARGET")

    print(
        f"No artificial color : "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Artificial color    : "
        f"{np.sum(y == 1)}"
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    groups = build_duplicate_groups(
        X_region
    )

    print()
    print("[2] DUPLICATES")

    print(
        f"Unique groups : "
        f"{len(np.unique(groups))}"
    )

    # --------------------------------------------------------
    # MANUAL LOGO KNN
    # --------------------------------------------------------

    print()
    print("[3] LEAVE-ONE-GROUP-OUT KNN")

    predictions = np.full(
        len(y),
        -1,
        dtype=int,
    )

    distances = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    unique_groups = np.unique(
        groups
    )

    for group in unique_groups:

        test_mask = (
            groups == group
        )

        train_mask = ~test_mask

        X_train = X_region[
            train_mask
        ]

        X_test = X_region[
            test_mask
        ]

        y_train = y[
            train_mask
        ]

        n_neighbors = min(
            3,
            len(X_train),
        )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights="distance",
                        metric="euclidean",
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_test
        )

        predictions[
            test_mask
        ] = pred

        if len(X_test) == 1:

            distances[
                test_mask
            ] = model.steps[
                1
            ][
                1
            ].kneighbors(
                model.steps[0][1].transform(
                    X_test
                ),
                n_neighbors=1,
                return_distance=True,
            )[0][:, 0]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions,
        )
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    print()
    print(
        f"Accuracy          : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        matrix
    )

    # --------------------------------------------------------
    # SAMPLE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "number":
                meta["number"],

            "name":
                meta["name"],

            "class":
                meta["class"],

            "true_artificial_color":
                y,

            "predicted_artificial_color":
                predictions,

            "nearest_distance":
                distances,

            "correct":
                y == predictions,
        }
    )

    results_path = (
        OUTPUT_DIR
        / "knn_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # DISTANCE RANKING
    # --------------------------------------------------------

    distance_results = (
        results
        .sort_values(
            "nearest_distance",
            ascending=False,
        )
    )

    distance_path = (
        OUTPUT_DIR
        / "distance_ranking.csv"
    )

    distance_results.to_csv(
        distance_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # REGION SUMMARY
    # --------------------------------------------------------

    region_summary = pd.DataFrame(
        {
            "ppm":
                ppm_region,

            "global_mean":
                X_region.mean(
                    axis=0
                ),

            "global_std":
                X_region.std(
                    axis=0
                ),
        }
    )

    region_summary_path = (
        OUTPUT_DIR
        / "region_statistics.csv"
    )

    region_summary.to_csv(
        region_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("KNN DIAGNOSTIC COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Predictions : {results_path}"
    )

    print(
        f"Distances   : {distance_path}"
    )

    print(
        f"Statistics  : {region_summary_path}"
    )


if __name__ == "__main__":
    main()