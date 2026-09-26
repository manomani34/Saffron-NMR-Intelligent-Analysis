# Saffron NMR — Robust/OOF alignment patch

## What changed

1. `Top-K=10` is no longer hard-coded in `origin_prediction.py`.
2. Robust Evaluation selects the Top-K using mean Balanced Accuracy.
3. Final OOF receives exactly that selected Top-K.
4. Robust and OOF use the same XGBoost hyperparameters.
5. Robust and OOF both use balanced `sample_weight` during training.
6. Robust and OOF both use mutual-information feature selection with `random_state=42`.
7. Robust evaluation uses explicit fold loops so training weights can be applied per fold.
8. OOF output records the actual `Model` and `TopK` used.

## Important scientific note

This fixes configuration inconsistency. It does **not** turn the current 43-sample / 11-class study into an operationally validated classifier.

Selecting the best Top-K on the same dataset and then reporting OOF from that selected configuration is still a model-selection-on-the-same-data procedure. A future stronger validation design would use nested cross-validation or an external test set.

## Main pipeline change

Run Robust Evaluation before final OOF, capture:

```python
robust_summary = evaluate_robust_models(...)
selected_top_k = int(robust_summary.iloc[0]["TopK"])
```

Then call:

```python
run_origin_prediction(
    engineered_features,
    y,
    metadata,
    top_k=selected_top_k,
    output_path="sample_predictions.csv",
)
```
