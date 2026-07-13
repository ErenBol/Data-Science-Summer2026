# Home Credit Default Risk - External Boosting Tuning Report

Run date: 2026-07-13

## Purpose

This experiment installs and compares external gradient boosting libraries on
the current best feature set:

`Curated raw + engineered`

Libraries tested:

- LightGBM `4.6.0`
- XGBoost `3.2.0`
- CatBoost `1.2.10`

The model family is selected on validation data. The best family is then tuned
with 8 sampled parameter configurations. The holdout test set is evaluated only
after selecting the tuned model and threshold.

## Baseline External Boosting Comparison

| Model | Threshold | Validation ROC-AUC | Validation AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM baseline | 0.680605 | 0.757775 | 0.238682 | 0.250000 | 0.393001 | 0.305599 |
| CatBoost baseline | 0.668477 | 0.756222 | 0.235139 | 0.238993 | 0.418177 | 0.304157 |
| XGBoost baseline | 0.666839 | 0.754955 | 0.234712 | 0.235236 | 0.418177 | 0.301097 |

LightGBM was the strongest external boosting family in the baseline comparison.

## LightGBM Tuning

The best validation-F1 configuration was:

```python
{
    "subsample": 1.0,
    "reg_lambda": 0.0,
    "num_leaves": 31,
    "n_estimators": 600,
    "min_child_samples": 50,
    "learning_rate": 0.02,
    "colsample_bytree": 0.85,
}
```

| Model | Threshold | Validation ROC-AUC | Validation AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM tuned 3 | 0.679819 | 0.759463 | 0.241413 | 0.248490 | 0.403827 | 0.307663 |
| LightGBM tuned 7 | 0.651243 | 0.758903 | 0.240978 | 0.235928 | 0.441088 | 0.307422 |
| LightGBM tuned 2 | 0.633791 | 0.758174 | 0.241084 | 0.230769 | 0.457704 | 0.306835 |
| LightGBM tuned 5 | 0.641017 | 0.759918 | 0.241508 | 0.228941 | 0.463243 | 0.306437 |

`LightGBM tuned 5` had the highest validation ROC-AUC, but `LightGBM tuned 3`
had the best validation F1, so `LightGBM tuned 3` was selected for the final
thresholded classifier.

## Holdout Test Performance

Selected model: `LightGBM tuned 3`

Selected threshold: `0.679819`

| Threshold Strategy | Threshold | ROC-AUC | AP | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.500000 | 0.764133 | 0.253209 | 0.709705 | 0.171952 | 0.680363 | 0.274523 |
| Validation selected | 0.679819 | 0.764133 | 0.253209 | 0.857405 | 0.257798 | 0.407855 | 0.315913 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,708 | 5,830 |
| Actual 1 | 2,940 | 2,025 |

## Comparison With Previous Best

Previous scikit-learn best from notebook 06:

- Model: Hist Gradient Boosting
- Test ROC-AUC: `0.762573`
- Test AP: `0.252267`
- Test F1: `0.316508`

New external boosting result:

- Model: LightGBM tuned 3
- Test ROC-AUC: `0.764133`
- Test AP: `0.253209`
- Test F1: `0.315913`

LightGBM is now the best ranking model by ROC-AUC and average precision. The
previous Hist Gradient Boosting model remains fractionally higher on holdout
F1, but the difference is very small and should not be treated as a meaningful
win without cross-validation.

## Recommendation

Use tuned LightGBM as the current best model for risk ranking. For production
classification decisions, keep threshold selection explicit:

- threshold `0.5` if high recall is the priority
- threshold `0.679819` if a more balanced precision/recall tradeoff is desired

Further gains are likely to require either broader tuning or aggregated
features from the relational Home Credit tables.
