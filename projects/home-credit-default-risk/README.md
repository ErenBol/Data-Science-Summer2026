# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Curated raw + engineered`

Notebook `05_feature_experimentation.ipynb` selected this feature set with:

- CV ROC-AUC: `0.745732`
- Holdout test ROC-AUC: `0.750681`
- Class 1 precision at threshold 0.5: `0.255421`
- Class 1 recall at threshold 0.5: `0.355891`
- Class 1 F1 at threshold 0.5: `0.297400`

The full raw application table underperformed the curated feature set, even
after adding engineered features. More columns were not better in this run.

## Project Notes

- `src/features.py` contains the shared curated feature list and deterministic
  row-level feature engineering.
- `src/preprocessing.py` builds numeric/categorical preprocessing pipelines.
- `src/evaluation.py` contains model and feature-set evaluation helpers.
- `reports/feature_experimentation_report.md` contains the latest executed
  feature experimentation report.

## Recommended Next Steps

- Keep `Curated raw + engineered` as the working feature set.
- Tune the probability threshold separately from model training.
- Try gradient boosting models before adding more raw columns.
- Add relational Home Credit tables only under the same CV/test protocol.
