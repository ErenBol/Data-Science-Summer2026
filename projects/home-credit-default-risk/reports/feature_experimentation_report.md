# Home Credit Default Risk - Feature Experimentation Report

Run date: 2026-07-13

## What Changed

- Centralized the curated application feature list in `src/features.py`.
- Made `create_features()` backward-compatible with smaller feature subsets.
- Added reusable feature-set CV evaluation in `src/evaluation.py`.
- Added optional numeric scaling in `src/preprocessing.py` for logistic regression.
- Reworked notebook `05_feature_experimentation.ipynb` into a clean feature-set experiment.
- Updated notebooks `03_mode_comparison.ipynb` and `04_model_tuning.ipynb` to use shared feature definitions.

## Notebook 03 - Model Comparison

Feature set: curated raw features plus deterministic engineered features.

| Model | Mean CV ROC-AUC | Std | Min | Max |
|---|---:|---:|---:|---:|
| Random Forest | 0.7443 | 0.0020 | 0.7420 | 0.7472 |
| Logistic Regression | 0.7417 | 0.0023 | 0.7394 | 0.7461 |
| Decision Tree | 0.7242 | 0.0017 | 0.7212 | 0.7261 |
| Dummy Classifier | 0.5000 | 0.0000 | 0.5000 | 0.5000 |

Random Forest remains the strongest model in this comparison, but logistic regression is close.

## Notebook 05 - Feature Set Comparison

| Feature Set | Raw Features | Transformed Features | Mean CV ROC-AUC | Std | Min | Max | Delta vs All Raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Curated raw + engineered, drop dominance >99.5% | 32 | 92 | 0.745732 | 0.001408 | 0.744304 | 0.747647 | 0.036610 |
| Curated raw + engineered | 32 | 92 | 0.745732 | 0.001408 | 0.744304 | 0.747647 | 0.036610 |
| Curated raw, drop dominance >99.5% | 17 | 63 | 0.740525 | 0.001585 | 0.739368 | 0.742767 | 0.031403 |
| Curated raw | 17 | 63 | 0.740525 | 0.001585 | 0.739368 | 0.742767 | 0.031403 |
| All raw + engineered, drop dominance >99.5% | 133 | 339 | 0.738945 | 0.001407 | 0.737362 | 0.740780 | 0.029823 |
| All raw + engineered | 149 | 355 | 0.738677 | 0.001085 | 0.737620 | 0.740168 | 0.029555 |
| All raw, drop dominance >99.5% | 104 | 289 | 0.712014 | 0.002534 | 0.708575 | 0.714606 | 0.002892 |
| All raw | 120 | 305 | 0.709122 | 0.002773 | 0.705262 | 0.711650 | 0.000000 |

Best feature set: `Curated raw + engineered`.

The dominance filter did not affect curated features because no curated columns crossed the 99.5% dominance threshold.

## Engineered Feature Ablation

Baseline: `Curated raw + engineered`, mean CV ROC-AUC `0.745732`.

| Removed Group | Mean CV ROC-AUC | Change vs Baseline | Removed Features |
|---|---:|---:|---:|
| financial | 0.739733 | -0.005999 | 4 |
| external | 0.743824 | -0.001908 | 5 |
| time | 0.744915 | -0.000817 | 3 |
| household | 0.745456 | -0.000276 | 4 |
| none | 0.745732 | 0.000000 | 0 |

Financial ratios are the most important engineered group in this run. External-source aggregations also help.

## Notebook 05 - Holdout Test Result

Selected feature set: `Curated raw + engineered, drop dominance >99.5%`

| Metric | Value |
|---|---:|
| Test ROC-AUC | 0.750681 |
| Precision, class 1 | 0.255421 |
| Recall, class 1 | 0.355891 |
| F1, class 1 | 0.297400 |
| Accuracy | 0.864251 |

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 51,387 | 5,151 |
| Actual 1 | 3,198 | 1,767 |

## Notebook 04 - Tuning Result

Tuning setup: 4 random Random Forest candidates, 3-fold CV, curated engineered feature set.

Best parameters:

```python
{
    "model__n_estimators": 200,
    "model__min_samples_split": 10,
    "model__min_samples_leaf": 5,
    "model__max_features": "log2",
    "model__max_depth": 12,
    "model__class_weight": "balanced_subsample",
}
```

Best tuning CV ROC-AUC: `0.744044`

Holdout test result:

| Metric | Value |
|---|---:|
| Test ROC-AUC | 0.748380 |
| Precision, class 1 | 0.181319 |
| Recall, class 1 | 0.591944 |
| F1, class 1 | 0.277605 |
| Accuracy | 0.751297 |

The tuned model improves recall at the default 0.5 threshold but lowers precision and accuracy. Its ROC-AUC is also slightly below the selected feature-set model from notebook 05.

## Recommendation

Use `Curated raw + engineered` as the current best feature set. Do not continue adding all raw application columns blindly; the full raw table underperformed the smaller curated set.

Next work:

- Tune threshold separately from model training.
- Try gradient boosting models such as LightGBM, XGBoost, or HistGradientBoosting.
- Add external relational tables only after preserving the same experiment protocol.
