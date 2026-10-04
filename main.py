import sys
import pandas as pd

sys.stdout.reconfigure(
    encoding="utf-8",
    errors="replace",
)

sys.stderr.reconfigure(
    encoding="utf-8",
    errors="replace",
)

from src.load_data import load_dataset
from src.validate_data import validate_dataset
from src.prepare_data import prepare_dataset
from src.year_analysis import analyze_harvest_year_effect
from src.feature_engineering import extract_spectral_features
from src.exploratory_analysis import run_pca
from src.novelty_detection import run_novelty_detection
from src.origin_prediction import run_origin_prediction
from src.stable_evaluation import evaluate_stable_models
from src.tree_models import evaluate_tree_models
from src.model_comparison import run_model_comparison
from src.robust_evaluation import evaluate_robust_models
from src.shap_analysis import run_shap_analysis
from src.decision_engine import run_decision_engine
from src.final_report import run_final_report


DATASET_PATH = "data/raw/saffron.csv"


def main():

    print("=" * 70)
    print("SAFFRON NMR PROJECT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_dataset(DATASET_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------
    # 2. Validation
    # --------------------------------------------------

    validate_dataset(df)

    # --------------------------------------------------
    # 3. Prepare dataset
    # --------------------------------------------------

    X, y, metadata = prepare_dataset(df)

    # --------------------------------------------------
    # 4. Harvest year effect analysis
    # --------------------------------------------------

    analyze_harvest_year_effect(
        X,
        y,
        metadata,
        output_dir="outputs",
    )

    # --------------------------------------------------
    # 5. PCA exploratory analysis
    # --------------------------------------------------

    run_pca(
        X,
        y,
        metadata,
        output_path="pca_scores.csv",
        plot_path="pca_scores.png",
        explained_variance_path="pca_explained_variance.csv",
        loadings_path="pca_loadings.csv",
        scree_plot_path="pca_scree_plot.png",
    )

    # --------------------------------------------------
    # 6. Novelty / OOD analysis
    # --------------------------------------------------

    run_novelty_detection(
        pca_scores_path="pca_scores.csv",
        explained_variance_path="pca_explained_variance.csv",
        output_path="novelty_detection_results.csv",
        plot_path="novelty_detection.png",
        confidence_level=0.99,
    )

    # --------------------------------------------------
    # 7. Feature engineering
    # --------------------------------------------------

    engineered_features = extract_spectral_features(X)

    engineered_features_output = engineered_features.copy()
    sample_ids = None

    if hasattr(metadata, "columns"):
        for column in [
            "SampleId",
            "sample_id",
            "name",
        ]:
            if column in metadata.columns:
                sample_ids = (
                    metadata[column]
                    .astype(str)
                    .reset_index(drop=True)
                )
                break

    elif isinstance(metadata, dict):
        for column in [
            "SampleId",
            "sample_id",
            "name",
        ]:
            if column in metadata:
                sample_ids = metadata[column]
                break

    if sample_ids is None:
        raise ValueError(
            "Could not find SampleId in metadata. "
            "Expected one of: SampleId, sample_id, name"
        )

    sample_ids = pd.Series(sample_ids).reset_index(drop=True)

    if len(sample_ids) != len(engineered_features_output):
        raise ValueError(
            "SampleId count does not match engineered feature rows. "
            f"SampleIds: {len(sample_ids)}, "
            f"Features: {len(engineered_features_output)}"
        )

    engineered_features_output.insert(
        0,
        "SampleId",
        sample_ids,
    )

    engineered_features_output.to_csv(
        "engineered_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nEngineered features saved to: "
        "engineered_features.csv"
    )

    print(
        f"Engineered feature rows: "
        f"{len(engineered_features_output)}"
    )

    print(
        f"Engineered feature columns: "
        f"{len(engineered_features_output.columns)}"
    )

    print("\nSampleId preview:")

    print(
        engineered_features_output[["SampleId"]]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 8. Robust XGBoost evaluation / configuration selection
    # --------------------------------------------------
    # Robust Evaluation is deliberately before final OOF.
    # The selected Top-K is passed into the final OOF path,
    # so the two paths cannot silently use different Top-K values.

    # --------------------------------------------------
    # Diagnostic: BEFORE Robust Evaluation
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA CHECK - BEFORE ROBUST EVALUATION")
    print("=" * 70)

    print(
        "engineered_features shape:",
        engineered_features.shape,
    )

    print(
        "y shape:",
        y.shape,
    )

    print("\nGroup distribution:")
    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nengineered_features index:")
    print(
        engineered_features.index.tolist()
    )

    print("\ny index:")
    print(
        y.index.tolist()
    )

    print("\nIndex aligned:")
    print(
        engineered_features.index.equals(
            y.index
        )
    )

    print("\nFeature columns count:")
    print(
        len(engineered_features.columns)
    )

    print("\nFirst 20 feature columns:")
    print(
        engineered_features.columns[:20].tolist()
    )

    print("\nFeature sample:")
    print(
        engineered_features.iloc[:3, :5]
    )

    # --------------------------------------------------
    # Robust Evaluation
    # --------------------------------------------------

    robust_summary = evaluate_robust_models(
        engineered_features,
        y,
        output_path="robust_evaluation_results.csv",
        summary_path="robust_evaluation_summary.csv",
    )

    # --------------------------------------------------
    # Diagnostic: AFTER Robust Evaluation
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA CHECK - AFTER ROBUST EVALUATION")
    print("=" * 70)

    print(
        "engineered_features shape:",
        engineered_features.shape,
    )

    print(
        "y shape:",
        y.shape,
    )

    print("\nGroup distribution:")
    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nengineered_features index:")
    print(
        engineered_features.index.tolist()
    )

    print("\ny index:")
    print(
        y.index.tolist()
    )

    print("\nIndex aligned:")
    print(
        engineered_features.index.equals(
            y.index
        )
    )

    print("\nFeature columns count:")
    print(
        len(engineered_features.columns)
    )

    print("\nFirst 20 feature columns:")
    print(
        engineered_features.columns[:20].tolist()
    )

    print("\nFeature sample:")
    print(
        engineered_features.iloc[:3, :5]
    )

    # --------------------------------------------------
    # Validate Robust result
    # --------------------------------------------------

    if robust_summary.empty:
        raise RuntimeError(
            "Robust evaluation returned no candidate configuration."
        )

    selected_top_k = int(
        robust_summary.iloc[0]["TopK"]
    )

    print("\n" + "=" * 70)
    print("FINAL ORIGIN CONFIGURATION")
    print("=" * 70)

    print("Model             : XGBoost")
    print(f"Selected Top-K    : {selected_top_k}")
    print("Selection source  : Robust Evaluation")

    # --------------------------------------------------
    # 9. Final OOF origin prediction
    # --------------------------------------------------

    run_origin_prediction(
        engineered_features,
        y,
        metadata,
        top_k=selected_top_k,
        output_path="sample_predictions.csv",
    )

    # --------------------------------------------------
    # 10. Stable model evaluation
    # --------------------------------------------------

    evaluate_stable_models(
        engineered_features,
        y,
        results_path="stable_model_results.csv",
    )

    # --------------------------------------------------
    # 11. Tree-based models
    # --------------------------------------------------

    evaluate_tree_models(
        engineered_features,
        y,
        results_path="tree_models_results.csv",
    )

    # --------------------------------------------------
    # 12. Model comparison
    # --------------------------------------------------

    run_model_comparison(
        output_path="model_comparison.csv"
    )

    # --------------------------------------------------
    # 13. SHAP feature importance
    # --------------------------------------------------
    # NOTE: For complete configuration consistency, the SHAP
    # module should also accept selected_top_k instead of using
    # an internal fixed Top-K. With the current Robust result
    # of Top-50 this happens to coincide, but the module should
    # be updated separately for future datasets.

    run_shap_analysis(
        engineered_features,
        y,
        output_path="shap_feature_importance.csv",
        plot_path="shap_summary.png",
    )

    # --------------------------------------------------
    # 14. Decision engine
    # --------------------------------------------------

    run_decision_engine(
        novelty_path="novelty_detection_results.csv",
        predictions_path="sample_predictions.csv",
        robust_path="robust_evaluation_summary.csv",
        output_path="decision_engine_results.csv",
    )

    # --------------------------------------------------
    # 15. Final research report
    # --------------------------------------------------

    run_final_report(
        output_path="final_report.csv"
    )

    print("\n" + "=" * 70)
    print("MAIN PIPELINE FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()