from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "reports/color_eda"
)

COLOR_REGION = (
    5.0,
    9.0,
)


# ============================================================
# HELPERS
# ============================================================

def load_data():
    X = np.load(X_PATH)

    ppm = np.load(PPM_PATH)

    meta = pd.read_csv(
        META_PATH
    )

    return X, ppm, meta


def region_mask(ppm, lower, upper):
    return (
        (ppm >= lower)
        & (ppm <= upper)
    )


def plot_all_spectra(
    X,
    ppm,
    meta,
):
    plt.figure(
        figsize=(14, 7)
    )

    for i in range(len(X)):
        label = (
            f"{meta.iloc[i]['number']} - "
            f"{meta.iloc[i]['name']}"
        )

        plt.plot(
            ppm,
            X[i],
            linewidth=0.8,
            alpha=0.7,
            label=label,
        )

    plt.xlabel(
        "Chemical shift (ppm)"
    )

    plt.ylabel(
        "Intensity"
    )

    plt.title(
        "All Color Dataset Spectra"
    )

    plt.gca().invert_xaxis()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "all_spectra.png",
        dpi=200,
    )

    plt.close()


def plot_color_region(
    X,
    ppm,
    meta,
):
    mask = region_mask(
        ppm,
        COLOR_REGION[0],
        COLOR_REGION[1],
    )

    plt.figure(
        figsize=(14, 7)
    )

    for i in range(len(X)):

        label = (
            f"{meta.iloc[i]['number']} - "
            f"{meta.iloc[i]['name']}"
        )

        plt.plot(
            ppm[mask],
            X[i][mask],
            linewidth=1.0,
            alpha=0.75,
            label=label,
        )

    plt.xlabel(
        "Chemical shift (ppm)"
    )

    plt.ylabel(
        "Intensity"
    )

    plt.title(
        "Artificial Color Region: 5–9 ppm"
    )

    plt.gca().invert_xaxis()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "color_region_5_9ppm.png",
        dpi=200,
    )

    plt.close()


def plot_real_vs_adulterated(
    X,
    ppm,
    meta,
):
    groups = [
        (
            "real_saffron",
            "Real saffron",
        ),

        (
            "saffron_plus_artificial_color",
            "Saffron + artificial color",
        ),

        (
            "artificial_color_adulterated_saffron",
            "Artificial-color adulterated",
        ),

        (
            "small_amount_artificial_color",
            "Small amount of color",
        ),

        (
            "unknown_adulterated_saffron",
            "Unknown adulteration",
        ),

        (
            "artificial_saffron",
            "Artificial saffron",
        ),
    ]

    for class_name, title in groups:

        mask_rows = (
            meta["class"]
            == class_name
        )

        indexes = np.where(
            mask_rows.to_numpy()
        )[0]

        if len(indexes) == 0:
            continue

        plt.figure(
            figsize=(14, 7)
        )

        for i in indexes:

            label = (
                f"{meta.iloc[i]['number']} - "
                f"{meta.iloc[i]['name']}"
            )

            plt.plot(
                ppm,
                X[i],
                linewidth=1.0,
                label=label,
            )

        plt.xlabel(
            "Chemical shift (ppm)"
        )

        plt.ylabel(
            "Intensity"
        )

        plt.title(
            title
        )

        plt.gca().invert_xaxis()

        plt.tight_layout()

        safe_name = (
            class_name
            .replace(
                " ",
                "_",
            )
        )

        plt.savefig(
            OUTPUT_DIR
            / f"{safe_name}.png",
            dpi=200,
        )

        plt.close()


def calculate_region_statistics(
    X,
    ppm,
    meta,
):
    mask = region_mask(
        ppm,
        COLOR_REGION[0],
        COLOR_REGION[1],
    )

    X_region = X[
        :,
        mask,
    ]

    rows = []

    for i in range(len(X_region)):

        spectrum = X_region[i]

        rows.append(
            {
                "number":
                    meta.iloc[i]["number"],

                "name":
                    meta.iloc[i]["name"],

                "class":
                    meta.iloc[i]["class"],

                "mean":
                    float(
                        np.mean(
                            spectrum
                        )
                    ),

                "std":
                    float(
                        np.std(
                            spectrum
                        )
                    ),

                "minimum":
                    float(
                        np.min(
                            spectrum
                        )
                    ),

                "maximum":
                    float(
                        np.max(
                            spectrum
                        )
                    ),

                "area":
    float(
        np.trapezoid(
            np.abs(spectrum),
            ppm[mask],
        )
    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COLOR DATASET EDA")
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
        f"X shape      : {X.shape}"
    )

    print(
        f"PPM shape    : {ppm.shape}"
    )

    print(
        f"Metadata rows: {len(meta)}"
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if X.shape[0] != len(meta):
        raise ValueError(
            "Number of spectra and metadata rows "
            "do not match."
        )

    if X.shape[1] != len(ppm):
        raise ValueError(
            "Number of spectral columns and PPM "
            "values do not match."
        )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print()
    print("[1] Plotting all spectra...")

    plot_all_spectra(
        X,
        ppm,
        meta,
    )

    print(
        "Saved: all_spectra.png"
    )

    print()
    print(
        "[2] Plotting 5–9 ppm region..."
    )

    plot_color_region(
        X,
        ppm,
        meta,
    )

    print(
        "Saved: color_region_5_9ppm.png"
    )

    print()
    print(
        "[3] Plotting class spectra..."
    )

    plot_real_vs_adulterated(
        X,
        ppm,
        meta,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print()
    print(
        "[4] Calculating 5–9 ppm statistics..."
    )

    statistics = calculate_region_statistics(
        X,
        ppm,
        meta,
    )

    statistics_path = (
        OUTPUT_DIR
        / "color_region_statistics.csv"
    )

    statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved: {statistics_path}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EDA COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()