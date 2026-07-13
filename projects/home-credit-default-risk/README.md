# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Top-200 gain-pruned relational aggregates`

`scripts/run_relational_feature_reduction.py` selects the smallest practical
model that keeps nearly the same ranking quality:

`Reduced relational LightGBM`

The selected classification threshold is:

`0.673881`

Feature count:

- Raw joined features: `200`
- Transformed model features: `244`

Holdout test metrics:

- ROC-AUC: `0.790318`
- Average precision: `0.287613`
- Class 1 precision: `0.278762`
- Class 1 recall: `0.449950`
- Class 1 F1: `0.344248`
- Accuracy: `0.861617`

The previous best application-table-only model from notebook `09` had ROC-AUC
`0.769069` and class-1 F1 `0.318450`. Adding full relational aggregate groups
improved ranking quality and the selected-threshold classifier. The full
relational model had `908` raw columns and `1884` transformed features. The
reduced model keeps nearly the same ROC-AUC with `200` raw columns and `244`
transformed features.

The best pure holdout ROC-AUC found so far is the compact 350-column model:
ROC-AUC `0.790435`, class-1 F1 `0.346246`, and `399` transformed features.
The top-200 model is preferred as the cleaner baseline because it is much
smaller with only a tiny ROC-AUC difference.

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
- `reports/external_boosting_tuning_report.md` contains the external boosting
  comparison and LightGBM tuning report.
- `reports/application_train_profile.html` contains the generated profiling
  report for `application_train.csv`.
- `reports/application_train_profile.json` contains the profiling report in
  JSON format for programmatic inspection.
- `reports/profile_guided_feature_pruning_report.md` contains the
  profile-guided feature-importance and pruning experiment.
- `reports/expanded_application_feature_groups_report.md` contains the
  expanded application feature group experiment.
- `reports/relational_profile_json_summary.md` summarizes the generated profile
  JSON files from all supplied tables.
- `reports/relational_feature_experiments_report.md` contains the relational
  feature group experiments, final metrics, and feature-importance findings.
- `reports/relational_feature_pruning_report.md` contains the relational
  pruning, grouped one-hot, and missing-indicator experiments.
- `reports/relational_feature_reduction_report.md` contains the stricter
  150/200/250-feature reduction experiments and model-save result.
- `requirements.txt` lists the project dependencies, including LightGBM,
  XGBoost, and CatBoost.
- `requirements-profiling.txt` lists the separate profiling dependency. The
  profiling report is generated in `.profiling-venv` because
  `ydata-profiling` currently requires `pandas<3`.

## Recommended Next Steps

- Use the top-200 reduced relational LightGBM model as the current clean
  baseline.
- Use threshold `0.673881` for the current balanced classifier, or threshold
  `0.5` when higher recall is more important than false positives.
- Tune LightGBM on the reduced relational matrix; the current hyperparameters
  were inherited from the application-table model.
- Add recent-history window aggregates, especially for bureau recency,
  installment lateness, previous refusals, and credit-card utilization.
- Compare tuned LightGBM against CatBoost on the same joined feature matrix.

## Profiling Report

Generate the profiling environment and report with:

```powershell
python -m venv .profiling-venv
.profiling-venv\Scripts\python.exe -m pip install -r requirements-profiling.txt
.profiling-venv\Scripts\python.exe scripts\generate_pandas_profile.py
.profiling-venv\Scripts\python.exe scripts\generate_relational_profiles.py --sample-rows 200000
```
