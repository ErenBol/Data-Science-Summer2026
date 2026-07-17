# Final Stacked Ensemble Report

Run date: 2026-07-17

## Setup

- Base models: advanced-pruned LightGBM and native-categorical CatBoost.
- Base-model out-of-fold predictions generated with the same 3-fold stratified splits as `run_advanced_relational_cv.py`.
- Meta-model: logistic regression on `lgbm_oof_pred`, `catboost_oof_pred`, plus top raw gain features.
- Current weighted-blend CV baseline: ROC-AUC `0.794531`, AP `0.290864`.

## Top Raw Features Used By Meta-Model

| feature |
| --- |
| EXT_SOURCE_MEAN |
| EXT_SOURCE_MIN |
| CREDIT_TERM_APPROX |
| EXT_SOURCE_MAX |
| BURO_BURO_DEBT_CREDIT_RATIO_MAX |
| INST_INST_LATE_PAYMENT_FLAG_MEAN |
| GOODS_CREDIT_RATIO |
| AGE_YEARS |
| PREV_DAYS_LAST_DUE_1ST_VERSION_MAX |
| PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN |

## Base Model OOF CV Diagnostics

| label | fold | lightgbm_transformed_feature_count | catboost_best_iteration | catboost_categorical_feature_count | lgbm_roc_auc | catboost_roc_auc | lgbm_average_precision | catboost_average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack_cv | 1 | 363 | 1199 | 14 | 0.792999 | 0.792618 | 0.293682 | 0.291553 |
| stack_cv | 2 | 363 | 1199 | 14 | 0.792493 | 0.793790 | 0.285731 | 0.286989 |
| stack_cv | 3 | 363 | 1199 | 14 | 0.792154 | 0.792765 | 0.284790 | 0.283059 |

## Meta-Model CV Results

| model | fold | roc_auc | average_precision |
| --- | --- | --- | --- |
| stacked_logistic_meta | 1 | 0.794572 | 0.292342 |
| stacked_logistic_meta | 2 | 0.794740 | 0.285814 |
| stacked_logistic_meta | 3 | 0.794539 | 0.285222 |

## CV Summary

| model | mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision |
| --- | --- | --- | --- | --- |
| current_50_50_weighted_blend | 0.794531 | nan | 0.290864 | nan |
| stacked_logistic_meta | 0.794617 | 0.000088 | 0.287793 | 0.003226 |

Delta versus current weighted blend:

| metric | delta |
| --- | --- |
| mean_cv_roc_auc | 0.000086 |
| mean_cv_average_precision | -0.003072 |

The stacked model improved CV ROC-AUC, so it was trained and evaluated on the holdout split.

Selected holdout threshold: `0.737425`

## Holdout Metrics

| model | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_stacked_ensemble | default_0.5 | 0.500000 | 0.796508 | 0.296714 | 0.718339 | 0.185739 | 0.735549 | 0.296585 |
| final_stacked_ensemble | validation_selected | 0.737425 | 0.796508 | 0.296714 | 0.866072 | 0.287145 | 0.444512 | 0.348905 |

## Holdout Comparison Against Current Native Blend

| model | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_native_catboost_blend | 0.668978 | 0.796126 | 0.299074 | 0.864722 | 0.286006 | 0.451561 | 0.350203 |
| final_stacked_ensemble | 0.737425 | 0.796508 | 0.296714 | 0.866072 | 0.287145 | 0.444512 | 0.348905 |

Delta, stacked model minus current native blend:

| metric | delta |
| --- | --- |
| threshold | 0.068447 |
| roc_auc | 0.000382 |
| average_precision | -0.002359 |
| accuracy | 0.001350 |
| precision_class_1 | 0.001140 |
| recall_class_1 | -0.007049 |
| f1_class_1 | -0.001298 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 51,059 | 5,479 |
| Actual 1 | 2,758 | 2,207 |

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

`C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_stacked_ensemble.joblib`
