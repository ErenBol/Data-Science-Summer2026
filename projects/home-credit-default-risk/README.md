# Home Credit Default Risk

Binary classification project for predicting default risk from the Home Credit
application dataset.

## Current Result

The current best feature set is:

`Curated raw + engineered`

Notebook `07_external_boosting_tuning.ipynb` selects the best current ranking
model:

`LightGBM tuned 3`

The selected classification threshold is:

`0.679819`

Holdout test metrics:

- ROC-AUC: `0.764133`
- Average precision: `0.253209`
- Class 1 precision: `0.257798`
- Class 1 recall: `0.407855`
- Class 1 F1: `0.315913`
- Accuracy: `0.857405`

The full raw application table underperformed the curated feature set, even
after adding engineered features. More columns were not better in this run.

The previous Hist Gradient Boosting model from notebook `06` has a fractionally
higher holdout F1 (`0.316508`), but tuned LightGBM has better ROC-AUC and
average precision. Use LightGBM for risk ranking and keep the threshold decision
explicit.

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
- `requirements.txt` lists the project dependencies, including LightGBM,
  XGBoost, and CatBoost.
- `requirements-profiling.txt` lists the separate profiling dependency. The
  profiling report is generated in `.profiling-venv` because
  `ydata-profiling` currently requires `pandas<3`.

## Recommended Next Steps

- Keep `Curated raw + engineered` as the working feature set.
- Use tuned LightGBM as the current best ranking model.
- Use threshold `0.679819` for the current balanced classifier, or threshold
  `0.5` when higher recall is more important than false positives.
- For a smaller ranking-focused model, use the 24-feature pruned LightGBM list
  from `08_profile_guided_feature_pruning.ipynb`. It slightly improves ROC-AUC
  but is slightly worse on F1.
- Run broader LightGBM tuning before changing the feature set.
- Add relational Home Credit tables only after model/threshold selection is
  stable under the same validation/test protocol.

## Profiling Report

Generate the profiling environment and report with:

```powershell
python -m venv .profiling-venv
.profiling-venv\Scripts\python.exe -m pip install -r requirements-profiling.txt
.profiling-venv\Scripts\python.exe scripts\generate_pandas_profile.py
```
