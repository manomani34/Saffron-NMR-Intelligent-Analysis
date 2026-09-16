from pathlib import Path
import pandas as pd


def _load_summary_file(
    path: str,
    model_name: str,
    configuration_column: str | None = None,
) -> pd.DataFrame | None:
    """
    Load a model summary CSV and convert it to
    a common comparison format.
    """

    file_path = Path(path)

    if not file_path.exists():
        print(f"WARNING: File not found -> {path}")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return None

    required_columns = [
        "AccuracyMean",
        "AccuracyStd",
        "BalancedAccuracyMean",
        "BalancedAccuracyStd",
        "F1MacroMean",
        "F1MacroStd",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(
            f"WARNING: Missing columns in {path}: "
            f"{missing_columns}"
        )
        return None

    rows = []

    for _, row in df.iterrows():

        if configuration_column is not None:
            configuration = str(
                row[configuration_column]
            )
        else:
            configuration = ""

        rows.append(
            {
                "Model": model_name,
                "Configuration": configuration,
                "AccuracyMean": float(
                    row["AccuracyMean"]
                ),
                "AccuracyStd": float(
                    row["AccuracyStd"]
                ),
                "BalancedAccuracyMean": float(
                    row["BalancedAccuracyMean"]
                ),
                "BalancedAccuracyStd": float(
                    row["BalancedAccuracyStd"]
                ),
                "F1MacroMean": float(
                    row["F1MacroMean"]
                ),
                "F1MacroStd": float(
                    row["F1MacroStd"]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_model_comparison(
    output_path: str = "model_comparison.csv",
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print("\n[1] LOADING MODEL RESULTS")

    sources = [
        (
            "svm_baseline_results.csv",
            "SVM",
            None,
        ),
        (
            "normalization_svm_results_summary.csv",
            "Normalization + SVM",
            "Experiment",
        ),
        (
            "pls_da_results_summary.csv",
            "PLS-DA",
            "Components",
        ),
        (
            "feature_svm_results_summary.csv",
            "Engineered Features + SVM",
            "TopK",
        ),
        (
            "tree_models_results_summary.csv",
            "Tree Models",
            "TopK",
        ),
    ]

    all_results = []

    for path, model_name, configuration_column in sources:

        loaded = _load_summary_file(
            path=path,
            model_name=model_name,
            configuration_column=configuration_column,
        )

        if loaded is not None:

            all_results.append(loaded)

            print(
                f"OK: Loaded {path}"
            )

    # ------------------------------------------------------
    # PCA + SVM
    # ------------------------------------------------------

    pca_svm_path = Path(
        "pca_svm_results.csv"
    )

    if pca_svm_path.exists():

        try:

            pca_df = pd.read_csv(
                pca_svm_path
            )

            required = [
                "Accuracy",
                "BalancedAccuracy",
                "F1Macro",
            ]

            if all(
                column in pca_df.columns
                for column in required
            ):

                pca_result = pd.DataFrame(
                    {
                        "Model": [
                            "PCA + SVM"
                        ],
                        "Configuration": [
                            "PCA 95%"
                        ],
                        "AccuracyMean": [
                            pca_df[
                                "Accuracy"
                            ].mean()
                        ],
                        "AccuracyStd": [
                            pca_df[
                                "Accuracy"
                            ].std()
                        ],
                        "BalancedAccuracyMean": [
                            pca_df[
                                "BalancedAccuracy"
                            ].mean()
                        ],
                        "BalancedAccuracyStd": [
                            pca_df[
                                "BalancedAccuracy"
                            ].std()
                        ],
                        "F1MacroMean": [
                            pca_df[
                                "F1Macro"
                            ].mean()
                        ],
                        "F1MacroStd": [
                            pca_df[
                                "F1Macro"
                            ].std()
                        ],
                    }
                )

                all_results.append(
                    pca_result
                )

                print(
                    "OK: Loaded pca_svm_results.csv"
                )

        except Exception as exc:

            print(
                "WARNING: Could not process "
                f"PCA + SVM: {exc}"
            )

    # ------------------------------------------------------
    # Check
    # ------------------------------------------------------

    if not all_results:

        raise FileNotFoundError(
            "No model result files were found."
        )

    comparison_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    comparison_df["Configuration"] = (
        comparison_df["Configuration"]
        .fillna("")
        .astype(str)
    )

    # ------------------------------------------------------
    # Ranking
    # ------------------------------------------------------
    # Primary criterion:
    # Balanced Accuracy
    #
    # Secondary:
    # Macro F1
    #
    # Tertiary:
    # Accuracy
    # ------------------------------------------------------

    comparison_df = comparison_df.sort_values(
        by=[
            "BalancedAccuracyMean",
            "F1MacroMean",
            "AccuracyMean",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    comparison_df.insert(
        0,
        "Rank",
        range(
            1,
            len(comparison_df) + 1,
        ),
    )

    # ------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] COMPLETE MODEL COMPARISON")
    print("=" * 70)

    display_columns = [
        "Rank",
        "Model",
        "Configuration",
        "AccuracyMean",
        "BalancedAccuracyMean",
        "F1MacroMean",
    ]

    print(
        comparison_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ------------------------------------------------------
    # Best candidate
    # ------------------------------------------------------

    best = comparison_df.iloc[0]

    print("\n" + "=" * 70)
    print("[3] CURRENT BEST CANDIDATE")
    print("=" * 70)

    print(
        f"Model             : "
        f"{best['Model']}"
    )

    print(
        f"Configuration     : "
        f"{best['Configuration']}"
    )

    print(
        f"Accuracy          : "
        f"{best['AccuracyMean']:.4f} "
        f"± {best['AccuracyStd']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{best['BalancedAccuracyMean']:.4f} "
        f"± {best['BalancedAccuracyStd']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{best['F1MacroMean']:.4f} "
        f"± {best['F1MacroStd']:.4f}"
    )

    # ------------------------------------------------------
    # Top 3
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] TOP 3 CANDIDATES")
    print("=" * 70)

    print(
        comparison_df.head(3)[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    comparison_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nComparison saved to: "
        f"{output_path}"
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON FINISHED")
    print("=" * 70)

    return comparison_df