import numpy as np
import pandas as pd


def extract_spectral_features(
    X: pd.DataFrame,
    window_size: float = 0.1,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("SPECTRAL FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Validate spectral axis
    # --------------------------------------------------

    try:
        ppm = np.asarray(
            [float(column) for column in X.columns],
            dtype=float,
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Spectral column names must be numeric ppm values."
        ) from exc

    if len(ppm) != X.shape[1]:
        raise ValueError(
            "Spectral axis length does not match X columns."
        )

    if len(ppm) < 2:
        raise ValueError(
            "At least two spectral points are required."
        )

    if not np.all(np.diff(ppm) > 0):
        raise ValueError(
            "Spectral axis must be strictly increasing."
        )

    print(f"Samples         : {X.shape[0]}")
    print(f"Spectral points : {X.shape[1]}")
    print(f"Window size     : {window_size} ppm")

    # --------------------------------------------------
    # 2. Convert to numpy
    # --------------------------------------------------

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input spectral matrix contains NaN or infinite values."
        )

    # --------------------------------------------------
    # 3. Global features
    # --------------------------------------------------

    print("\n[1] GLOBAL FEATURES")

    feature_data = {}

    feature_data["Global_Mean"] = np.mean(values, axis=1)
    feature_data["Global_Std"] = np.std(values, axis=1)
    feature_data["Global_Min"] = np.min(values, axis=1)
    feature_data["Global_Max"] = np.max(values, axis=1)
    feature_data["Global_Median"] = np.median(values, axis=1)
    feature_data["Global_L1"] = np.sum(np.abs(values), axis=1)
    feature_data["Global_L2"] = np.linalg.norm(values, axis=1)

    for quantile in [
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]:
        feature_data[
            f"Global_Q{int(quantile * 100)}"
        ] = np.quantile(
            values,
            quantile,
            axis=1,
        )

    global_feature_count = len(feature_data)

    print(
        f"Generated global features: "
        f"{global_feature_count}"
    )

    # --------------------------------------------------
    # 4. Spectral windows
    # --------------------------------------------------

    print("\n[2] SPECTRAL WINDOWS")

    min_ppm = float(np.min(ppm))
    max_ppm = float(np.max(ppm))

    current_start = min_ppm
    window_count = 0

    while current_start < max_ppm:

        current_end = current_start + window_size

        mask = (
            (ppm >= current_start)
            & (ppm < current_end)
        )

        if np.any(mask):

            window_values = values[:, mask]

            prefix = (
                f"Window_"
                f"{current_start:.3f}_"
                f"{current_end:.3f}"
            )

            feature_data[
                f"{prefix}_Mean"
            ] = np.mean(
                window_values,
                axis=1,
            )

            feature_data[
                f"{prefix}_Std"
            ] = np.std(
                window_values,
                axis=1,
            )

            feature_data[
                f"{prefix}_Max"
            ] = np.max(
                window_values,
                axis=1,
            )

            feature_data[
                f"{prefix}_Sum"
            ] = np.sum(
                window_values,
                axis=1,
            )

            window_count += 1

        current_start = current_end

    print(
        f"Generated spectral windows: "
        f"{window_count}"
    )

    # --------------------------------------------------
    # 5. Create feature matrix
    # --------------------------------------------------

    features = pd.DataFrame(
        feature_data,
        index=X.index,
    )

    # --------------------------------------------------
    # 6. Validation
    # --------------------------------------------------

    if features.empty:
        raise ValueError(
            "Feature engineering generated an empty feature matrix."
        )

    if features.isna().any().any():
        raise ValueError(
            "Feature engineering generated NaN values."
        )

    if not np.isfinite(
        features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Feature engineering generated infinite values."
        )

    # --------------------------------------------------
    # 7. Final information
    # --------------------------------------------------

    print("\n[3] FINAL FEATURE MATRIX")

    print(
        f"Original dimensions : {X.shape}"
    )

    print(
        f"Feature dimensions  : {features.shape}"
    )

    print(
        f"Compression ratio   : "
        f"{X.shape[1] / features.shape[1]:.2f}"
    )

    print(
        f"Feature count       : "
        f"{features.shape[1]}"
    )

    print(
        "Feature preview     : "
        "first 3 rows / first 5 features only"
    )

    print(
        features.iloc[
            :3,
            :5,
        ].round(4).to_string(index=False)
    )

    if features.shape[1] > 5:
        print(
            f"... {features.shape[1] - 5} "
            f"additional features omitted."
        )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING FINISHED")
    print("=" * 70)

    return features