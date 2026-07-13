# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Top-200 gain-pruned relational aggregates`

`scripts/run_reduced_model_selection.py` compares CatBoost against the reduced
feature baseline and tunes LightGBM with Optuna when CatBoost does not improve
validation ROC-AUC enough:

`Optuna-tuned reduced relational LightGBM`

The selected classification threshold is:

`0.656411`

Feature count:

- Raw joined features: `200`
- Transformed model features: `244`

Holdout test metrics:

- ROC-AUC: `0.791323`
- Average precision: `0.290276`
- Class 1 precision: `0.275915`
- Class 1 recall: `0.467472`
- Class 1 F1: `0.347014`
- Accuracy: `0.857974`

The previous best application-table-only model from notebook `09` had ROC-AUC
`0.769069` and class-1 F1 `0.318450`. Adding full relational aggregate groups
improved ranking quality and the selected-threshold classifier. The full
relational model had `908` raw columns and `1884` transformed features. The
reduced model keeps nearly the same ROC-AUC with `200` raw columns and `244`
transformed features.

CatBoost on the same reduced 200-feature set reached validation ROC-AUC
`0.787485`, which was below the previous reduced LightGBM validation ROC-AUC
`0.787824`, so LightGBM was tuned with Optuna instead of tuning CatBoost.

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
- `reports/reduced_model_selection_report.md` contains the CatBoost comparison
  and Optuna-tuned LightGBM result.
- `requirements.txt` lists the project dependencies, including LightGBM,
  XGBoost, and CatBoost.
- `requirements-profiling.txt` lists the separate profiling dependency. The
  profiling report is generated in `.profiling-venv` because
  `ydata-profiling` currently requires `pandas<3`.

## Recommended Next Steps

- Use the Optuna-tuned top-200 reduced relational LightGBM model as the current
  best model.
- Use threshold `0.656411` for the current balanced classifier, or threshold
  `0.5` when higher recall is more important than false positives.
- Add recent-history window aggregates; tuning alone did not reach `0.80`
  ROC-AUC.
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
