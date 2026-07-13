# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Curated raw + engineered`

Notebook `06_model_threshold_selection.ipynb` selects the best current model:

`Hist Gradient Boosting`

The selected classification threshold is:

`0.658010`

Holdout test metrics:

- ROC-AUC: `0.762573`
- Average precision: `0.252267`
- Class 1 precision: `0.243290`
- Class 1 recall: `0.452769`
- Class 1 F1: `0.316508`
- Accuracy: `0.842138`

The full raw application table underperformed the curated feature set, even
after adding engineered features. More columns were not better in this run.

## Project Notes

- `src/features.py` contains the shared curated feature list and deterministic
  row-level feature engineering.
- `src/preprocessing.py` builds numeric/categorical preprocessing pipelines.
- `src/evaluation.py` contains model and feature-set evaluation helpers.
- `src/thresholding.py` contains probability and threshold evaluation helpers.
- `reports/feature_experimentation_report.md` contains the latest executed
  feature experimentation report.
- `reports/model_threshold_selection_report.md` contains the latest model and
  threshold selection report.

## Recommended Next Steps

- Keep `Curated raw + engineered` as the working feature set.
- Use Hist Gradient Boosting as the current best local model.
- Install and test LightGBM, XGBoost, or CatBoost for the next model jump.
- Add relational Home Credit tables only after model/threshold selection is
  stable under the same validation/test protocol.
