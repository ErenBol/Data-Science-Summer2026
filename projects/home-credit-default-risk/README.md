# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Curated raw + engineered`

Notebook `09_expanded_application_feature_groups.ipynb` selects the best
current application-table model:

`Expanded LightGBM`

The selected classification threshold is:

`0.688844`

Holdout test metrics:

- ROC-AUC: `0.769069`
- Average precision: `0.258392`
- Class 1 precision: `0.268735`
- Class 1 recall: `0.390735`
- Class 1 F1: `0.318450`
- Accuracy: `0.864982`

The full raw application table underperformed the curated feature set, even
after adding engineered features. More columns were not better in this run.

The expanded LightGBM model improves ROC-AUC, average precision, and class-1 F1
over the previous tuned LightGBM setup from notebook `07`.

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
- `requirements.txt` lists the project dependencies, including LightGBM,
  XGBoost, and CatBoost.
- `requirements-profiling.txt` lists the separate profiling dependency. The
  profiling report is generated in `.profiling-venv` because
  `ydata-profiling` currently requires `pandas<3`.

## Recommended Next Steps

- Use expanded application features as the current best feature set.
- Use expanded LightGBM as the current best application-table model.
- Use threshold `0.688844` for the current balanced classifier, or threshold
  `0.5` when higher recall is more important than false positives.
- For a smaller ranking-focused model, use the 24-feature pruned LightGBM list
  from `08_profile_guided_feature_pruning.ipynb`. It slightly improves ROC-AUC
  but is slightly worse on F1.
- Run broader LightGBM tuning on the expanded feature set.
- Add relational Home Credit tables only after model/threshold selection is
  stable under the same validation/test protocol.

## Profiling Report

Generate the profiling environment and report with:

```powershell
python -m venv .profiling-venv
.profiling-venv\Scripts\python.exe -m pip install -r requirements-profiling.txt
.profiling-venv\Scripts\python.exe scripts\generate_pandas_profile.py
```
