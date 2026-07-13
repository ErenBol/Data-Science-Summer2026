# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Expanded application features + all relational table aggregates`

Notebook `10_relational_feature_experiments.ipynb` and
`scripts/run_relational_feature_experiments.py` select the best current model:

`Relational LightGBM`

The selected classification threshold is:

`0.687520`

Holdout test metrics:

- ROC-AUC: `0.790203`
- Average precision: `0.288818`
- Class 1 precision: `0.289514`
- Class 1 recall: `0.423162`
- Class 1 F1: `0.343806`
- Accuracy: `0.869600`

The previous best application-table-only model from notebook `09` had ROC-AUC
`0.769069` and class-1 F1 `0.318450`. Adding full relational aggregate groups
improved ranking quality and the selected-threshold classifier.

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
- `requirements.txt` lists the project dependencies, including LightGBM,
  XGBoost, and CatBoost.
- `requirements-profiling.txt` lists the separate profiling dependency. The
  profiling report is generated in `.profiling-venv` because
  `ydata-profiling` currently requires `pandas<3`.

## Recommended Next Steps

- Use the relational LightGBM model as the current best model.
- Use threshold `0.687520` for the current balanced classifier, or threshold
  `0.5` when higher recall is more important than false positives.
- Tune LightGBM on the full relational matrix; the current hyperparameters were
  inherited from the application-table model.
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
