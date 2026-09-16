# Saffron NMR Intelligent Analysis System

## Research Title

**Intelligent System for Determining Geographical Origin, Assessing Authenticity, and Identifying Unknown and Suspicious Saffron Samples Based on NMR Spectroscopic Data**

## 1. Project Overview

This project develops a reproducible computational framework for analyzing saffron NMR spectral data with three main objectives:

1. Exploration of geographical origin discrimination.
2. Assessment of sample novelty and out-of-domain (OOD) behavior.
3. Development of a framework for future authenticity and adulteration assessment.

The current study is a pilot/proof-of-concept analysis. The objective is to evaluate the feasibility of the proposed analytical framework and identify the data requirements needed for a more reliable future model.

---

## 2. Current Dataset

The current dataset contains:

* 44 original records.
* 43 independent observations after removal of one confirmed duplicate measurement.
* 11 geographic groups.
* 9 regions represented in harvest year 1394.
* 9 regions represented in harvest year 1404.
* 7 regions represented in both years.
* 9,514 spectral variables per independent sample.

The dataset contains no missing spectral values and all spectral values are numeric.

One duplicate measurement was identified:

* Original sample name: `94-30`
* Sample IDs: `34` and `35`
* Sample ID `35` was removed from modeling.

---

## 3. Spectral Preprocessing

The supplied spectra were processed externally in Mnova.

Current preprocessing information:

* Baseline correction: Applied in Mnova.
* Reference/alignment: Applied in Mnova.
* Binning: Fixed-width.
* Nominal bin width: 0.001 ppm.
* Aggregation: Sum.
* Python-stage normalization: Not applied.

The spectral range is approximately 0.5–10.5 ppm.

Three known removed spectral regions are preserved as gaps in the spectral axis.

---

## 4. Analysis Pipeline

The final pipeline consists of:

```text
Load Dataset
    ↓
Dataset Validation
    ↓
Dataset Preparation
    ↓
Harvest Year Effect Analysis
    ↓
PCA Exploratory Analysis
    ↓
Mahalanobis OOD / Novelty Detection
    ↓
Spectral Feature Engineering
    ↓
Origin OOF Diagnostic
    ↓
Robust Origin Model Evaluation
    ↓
SHAP Model Interpretation
    ↓
Decision Engine
    ↓
Final Research Report
```

---

## 5. Harvest Year Analysis

Harvest-year analysis evaluates whether the NMR spectra contain information related to harvest year and whether geographic origin transfers between harvest years.

The current data contain:

* 18 independent samples from 1394.
* 25 independent samples from 1404.

Cross-validated prediction of harvest year produced:

* Accuracy: approximately 0.72.
* Balanced Accuracy: approximately 0.71.
* Macro F1: approximately 0.69.

This indicates that the spectral data contain detectable year-related structure.

However, the within-region permutation test produced:

* p-value = 0.1704.

Therefore, the current data do not provide sufficient evidence to establish a statistically significant independent within-region harvest-year effect.

---

## 6. PCA

PCA is used only as an exploratory unsupervised analysis.

Current explained variance:

* PC1: approximately 50.25%.
* PC2: approximately 23.06%.
* PC3: approximately 9.13%.
* 5 PCs explain approximately 90% of variance.
* 8 PCs explain approximately 95.5% of variance.

PCA results are not interpreted as classification performance.

---

## 7. Geographic Origin Modeling

Several machine-learning models and feature configurations were evaluated.

The final reference candidate is selected from repeated robust evaluation rather than from a single train/test split.

Current best robust candidate:

* Model: XGBoost.
* Selected features: Top-50.
* Accuracy: approximately 0.19 ± 0.05.
* Balanced Accuracy: approximately 0.12 ± 0.04.
* Macro F1: approximately 0.10 ± 0.04.
* Chance Balanced Accuracy for 11 classes: approximately 0.091.

Although the model is slightly above chance level, its performance is not sufficiently strong or stable for operational geographic-origin prediction.

Therefore:

**Origin model status: NOT RELIABLE**

---

## 8. Cross-Year Geographic Generalization

Cross-year validation was performed between the two harvest years using only the seven regions represented in both years.

Best current cross-year result:

* Direction: 1404 → 1394.
* Model: XGBoost.
* Top-K: 30.
* Accuracy: approximately 0.21.
* Balanced Accuracy: approximately 0.19.
* Macro F1: approximately 0.14.

These results are strongly constrained by the small number of samples and the fact that several geographic classes contain only one training sample in one direction.

Therefore, cross-year geographic generalization is considered weak and inconclusive.

---

## 9. Novelty / OOD Detection

Mahalanobis distance in PCA space is used to identify observations that are outside the learned reference domain.

Using eight PCs and a 99% confidence threshold:

* 42 samples were within the reference domain.
* 1 sample was identified as a novelty/OOD candidate.

Current OOD candidate:

* Sample ID: 20.
* Group: G5.

Important:

**OOD does not mean adulterated.**

The detected sample should be considered a candidate for expert review rather than proof of adulteration.

---

## 10. SHAP Interpretation

SHAP is used to identify features that contribute most strongly to the final XGBoost reference model.

The selected model uses the Top-50 feature configuration.

The resulting spectral windows should be interpreted as:

* model-associated spectral regions,
* candidate regions for scientific investigation,
* not validated biomarkers.

Because the current origin model has limited performance and the dataset is small, SHAP findings should not be interpreted as definitive chemical markers of geographic origin.

---

## 11. Authenticity and Adulteration

The current dataset is not sufficient for a validated adulteration classifier.

The current system therefore reports:

**NOT ASSESSED / REQUIRES EXPERT REVIEW**

Future authenticity modeling requires validated samples such as:

* authentic reference samples,
* adulterated samples,
* mixed samples,
* type of adulterant,
* mixing percentage,
* and reliable laboratory confirmation.

---

## 12. Main Scientific Findings

The current pilot demonstrates that:

1. A reproducible NMR data-processing and machine-learning pipeline can be implemented.
2. The spectra contain detectable harvest-year-related structure.
3. Geographic-origin discrimination is currently weak and not operationally reliable.
4. Cross-year geographic generalization is weak under the current sample design.
5. Mahalanobis-based novelty detection can identify candidate samples outside the reference domain.
6. OOD detection must not be interpreted as proof of adulteration.
7. SHAP can identify model-associated spectral regions for further investigation.
8. The main limitation is the current sample size and distribution rather than the absence of a computational framework.

---

## 13. Main Data Limitations

The current dataset contains only 43 independent observations across 11 geographic classes.

Several regions contain only two samples, and some regions occur in only one harvest year.

This creates:

* strong class imbalance,
* limited within-region replication,
* incomplete Region × Year coverage,
* unstable model estimation,
* and limited cross-year generalization.

These limitations prevent the current dataset from supporting a production-grade geographic-origin model.

---

## 14. Recommended Future Data

Future data collection should prioritize:

### Geographic replication

Increase the number of independent samples per region.

### Multi-year coverage

Collect samples from each region across multiple harvest years.

### Complete metadata

For every sample, record where available:

* unique sample ID,
* exact geographic location,
* harvest year,
* harvest date,
* cultivation conditions,
* cultivar,
* processing method,
* drying method,
* storage conditions,
* NMR sample preparation,
* NMR acquisition parameters.

### Authenticity reference set

Add validated:

* authentic samples,
* adulterated samples,
* mixed samples,
* adulterant identity,
* mixing percentage,
* laboratory confirmation.

---

## 15. Scientific Interpretation Policy

The project follows a conservative interpretation policy:

* Low classification performance is reported honestly.
* Above-chance performance is not automatically labeled reliable.
* PCA separation is not treated as classification performance.
* OOD is not treated as proof of adulteration.
* SHAP features are not treated as validated biomarkers.
* Authenticity is not claimed without validated adulterated reference data.

---

## 16. Project Status

**Current status: Pilot / Proof-of-Concept**

The computational framework is operational and reproducible.

The current dataset is suitable for:

* exploratory analysis,
* methodology development,
* OOD screening,
* hypothesis generation,
* and identification of future data requirements.

The current dataset is not yet sufficient for:

* production-grade geographic-origin prediction,
* validated authenticity classification,
* or definitive adulteration detection.


## نتیجه‌گیری پژوهش

[مشاهده نتیجه‌گیری نهایی فارسی](docs/conclusion_fa.md)