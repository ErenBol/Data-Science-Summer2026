# Home Credit Default Risk - Model and Threshold Selection Report

Run date: 2026-07-13

## Purpose

This experiment keeps the current best feature set from notebook 05:

`Curated raw + engineered`

It then compares available scikit-learn models and selects a classification
threshold using a validation split from the training data. The holdout test set
is evaluated only after selecting the model and threshold.

External libraries such as LightGBM, XGBoost, and CatBoost were not installed in
the current environment, so this pass uses scikit-learn models only.

## Data Split

| Split | Rows | Default Rate |
|---|---:|---:|
| Train | 196,806 | 0.080729 |
| Validation | 49,202 | 0.080728 |
| Test | 61,503 | 0.080728 |

Feature matrix: 32 curated raw plus engineered features.

## Validation Model Comparison

Models were ranked by best validation F1 after threshold selection, with ROC-AUC
used as supporting ranking quality.

| Model | Validation ROC-AUC | Validation AP | Default F1 | Best Threshold | Precision | Recall | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hist Gradient Boosting | 0.757085 | 0.237036 | 0.269402 | 0.658010 | 0.233276 | 0.443353 | 0.305703 |
| Random Forest current | 0.743612 | 0.222539 | 0.289260 | 0.444572 | 0.217898 | 0.446878 | 0.292953 |
| Random Forest tuned shallow | 0.742576 | 0.217561 | 0.276678 | 0.574532 | 0.216952 | 0.437563 | 0.290078 |
| Logistic Regression | 0.741156 | 0.217104 | 0.254009 | 0.663440 | 0.223953 | 0.405337 | 0.288505 |
| Extra Trees | 0.739272 | 0.215776 | 0.272604 | 0.566649 | 0.211746 | 0.439325 | 0.285761 |

Selected model: `Hist Gradient Boosting`

Selected threshold: `0.658010`

## Holdout Test Performance

| Threshold Strategy | Threshold | ROC-AUC | AP | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.500000 | 0.762573 | 0.252267 | 0.706990 | 0.169870 | 0.676536 | 0.271555 |
| Validation best F1 | 0.658010 | 0.762573 | 0.252267 | 0.842138 | 0.243290 | 0.452769 | 0.316508 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 49,546 | 6,992 |
| Actual 1 | 2,717 | 2,248 |

## Interpretation

Hist Gradient Boosting is now the best model in the current setup. It improves
holdout ROC-AUC from the previous Random Forest result of `0.750681` to
`0.762573`.

The tuned threshold of `0.658010` improves class-1 F1 from `0.271555` at the
default threshold to `0.316508`. It also gives a more balanced operating point:

- fewer false positives than threshold `0.5`
- lower recall than threshold `0.5`
- better precision and F1 than threshold `0.5`

If the goal is catching as many defaults as possible, threshold `0.5` may still
be useful because recall is `0.676536`. If the goal is a more balanced default
flag, use threshold `0.658010`.

## Next Performance Improvements

The current application-only dataset is close to its ceiling for simple
scikit-learn models. The next meaningful performance jump will likely require
one or both of these:

- Install and test LightGBM, XGBoost, or CatBoost.
- Add aggregated features from the relational Home Credit datasets:
  `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`,
  `installments_payments.csv`, `credit_card_balance.csv`, and
  `POS_CASH_balance.csv`.

For now, the best next local step is external gradient boosting libraries before
adding the relational tables.
