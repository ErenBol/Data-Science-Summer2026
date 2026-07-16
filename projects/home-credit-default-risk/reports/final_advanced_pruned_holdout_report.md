# Final Advanced Pruned Holdout Report

Run date: 2026-07-16

## Setup

- Same train/holdout split as the reduced-model experiments.
- Final feature recipe: top-200 reduced base + recent/domain relational features + fold-safe target encodings.
- Dropped transformed features:
  - `categorical__OCCUPATION_TYPE_infrequent_sklearn`
  - `categorical__OCCUPATION_TYPE_Security staff`
- Selected transformed feature count: `364`
- Threshold selected on validation split: `0.670438`

## Holdout Metrics

| model | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_advanced_pruned_lgbm | default_0.5 | 0.500000 | 0.794353 | 0.296705 | 0.759085 | 0.203574 | 0.681370 | 0.313487 |
| final_advanced_pruned_lgbm | validation_selected | 0.670438 | 0.794353 | 0.296705 | 0.867112 | 0.289391 | 0.443907 | 0.350370 |

## Comparison Against Previous Saved Reduced Optuna Model

| model | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_reduced_lightgbm_optuna | 0.656411 | 0.791323 | 0.290276 | 0.857974 | 0.275915 | 0.467472 | 0.347014 |
| final_advanced_pruned_lgbm | 0.670438 | 0.794353 | 0.296705 | 0.867112 | 0.289391 | 0.443907 | 0.350370 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 51,126 | 5,412 |
| Actual 1 | 2,761 | 2,204 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.90      0.93     56538
           1       0.29      0.44      0.35      4965

    accuracy                           0.87     61503
   macro avg       0.62      0.67      0.64     61503
weighted avg       0.90      0.87      0.88     61503

```

Saved model bundle:

`C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_advanced_pruned_lgbm.joblib`
