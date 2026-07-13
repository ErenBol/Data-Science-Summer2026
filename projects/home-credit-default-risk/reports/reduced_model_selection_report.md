# Reduced Feature Model Selection Report

Run date: 2026-07-14

## Decision Rule

CatBoost had to beat the reduced LightGBM validation ROC-AUC by at least `0.002` to justify tuning CatBoost. Otherwise, Optuna tunes LightGBM on the same top-200 reduced feature set.

## Validation Comparison

| model | phase | raw_feature_count | transformed_feature_count | threshold | best_iteration | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_reduced_previous | validation | 200 | 244 | 0.673881 |  | 0.787824 | 0.272900 | 0.857546 | 0.268062 | 0.441843 | 0.333682 |
| catboost_baseline | validation | 200 | 200 | 0.717247 | 1199 | 0.787485 | 0.271535 | 0.877871 | 0.300958 | 0.387714 | 0.338871 |
| lightgbm_optuna | validation | 200 | 244 | 0.656411 | 1008 | 0.789241 | 0.276490 | 0.857526 | 0.271235 | 0.453424 | 0.339427 |

Selected path: `lightgbm_optuna`

## Final Holdout Metrics

| model | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_optuna_final | default_0.5 | 0.500000 | 0.791323 | 0.290276 | 0.755508 | 0.200595 | 0.679557 | 0.309754 |
| lightgbm_optuna_final | validation_selected | 0.656411 | 0.791323 | 0.290276 | 0.857974 | 0.275915 | 0.467472 | 0.347014 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,447 | 6,091 |
| Actual 1 | 2,644 | 2,321 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.89      0.92     56538
           1       0.28      0.47      0.35      4965

    accuracy                           0.86     61503
   macro avg       0.61      0.68      0.63     61503
weighted avg       0.90      0.86      0.87     61503

```

## Best Parameters

```python
{'n_estimators': 1008, 'learning_rate': 0.02178196621205854, 'num_leaves': 32, 'max_depth': 9, 'min_child_samples': 179, 'subsample': 0.9978522016969403, 'colsample_bytree': 0.8265635267137263, 'reg_alpha': 1.224310860385835e-08, 'reg_lambda': 2.6072505559148544e-06, 'min_split_gain': 0.514970414020943}
```

## Saved Model

`C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_reduced_lightgbm_optuna.joblib`

## Final LightGBM Feature Importance

| transformed_feature | gain_importance |
| --- | --- |
| numeric__EXT_SOURCE_MEAN | 643932.606860 |
| numeric__EXT_SOURCE_MIN | 129140.529750 |
| numeric__EXT_SOURCE_MAX | 91619.144792 |
| numeric__CREDIT_TERM_APPROX | 72712.792327 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MAX | 56978.679995 |
| numeric__INST_INST_LATE_PAYMENT_FLAG_MEAN | 47992.850400 |
| numeric__GOODS_CREDIT_RATIO | 44096.794520 |
| numeric__AGE_YEARS | 42405.794330 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_MAX | 36261.741364 |
| numeric__AMT_ANNUITY | 31791.587298 |
| numeric__PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN | 30891.530443 |
| numeric__INST_AMT_PAYMENT_SUM | 22833.940948 |
| numeric__EXT_SOURCE_3 | 22198.511652 |
| numeric__PREV_NAME_CONTRACT_STATUS_REFUSED_RATE | 21147.644858 |
| categorical__NAME_EDUCATION_TYPE_Higher education | 20886.916624 |
| numeric__CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | 20070.831664 |
| numeric__DAYS_EMPLOYED | 19852.697005 |
| numeric__AMT_GOODS_PRICE | 18802.987383 |
| numeric__EMPLOYMENT_AGE_RATIO | 18336.096251 |
| numeric__EXT_SOURCE_1 | 17716.167669 |
| numeric__ANNUITY_INCOME_RATIO | 17413.806856 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MEAN | 17352.722264 |
| categorical__CODE_GENDER_F | 16752.590530 |
| numeric__EMPLOYED_YEARS | 16603.917791 |
| numeric__EXT_SOURCE_2 | 15847.657590 |
| numeric__AMT_CREDIT | 15384.196668 |
| categorical__CODE_GENDER_M | 13920.143665 |
| numeric__EXT_SOURCE_COUNT | 13562.407042 |
| numeric__BURO_DAYS_CREDIT_ENDDATE_MAX | 13337.425580 |
| numeric__CC_CC_UTILIZATION_RATIO_MAX | 13288.826122 |
| numeric__PREV_DAYS_LAST_DUE_MAX | 13210.081171 |
| numeric__ANNUITY_PER_PERSON | 13077.088102 |
| numeric__INST_AMT_PAYMENT_MIN | 12854.500065 |
| numeric__INST_AMT_PAYMENT_MEAN | 12658.885040 |
| numeric__POS_POS_COMPLETION_RATIO_SUM | 12457.499353 |
| numeric__OWN_CAR_AGE | 12186.944593 |
| numeric__BURO_DAYS_ENDDATE_FACT_MAX | 12186.072040 |
| numeric__EXT_SOURCE_STD | 12083.458763 |
| numeric__CREDIT_PER_PERSON | 11907.556925 |
| numeric__CREDIT_INCOME_RATIO | 11904.870896 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_SUM | 11541.147758 |
| numeric__INST_INST_PAYMENT_DELAY_MAX | 11343.708385 |
| numeric__BURO_AMT_CREDIT_SUM_SUM | 10659.491202 |
| numeric__BURO_AMT_CREDIT_SUM_MAX | 10589.788381 |
| numeric__POS_POS_COMPLETION_RATIO_MEAN | 10493.743320 |
| numeric__BURO_DAYS_CREDIT_SUM | 10025.719613 |
| numeric__PREV_AMT_DOWN_PAYMENT_SUM | 9948.919504 |
| numeric__REGION_POPULATION_RELATIVE | 9828.700085 |
| numeric__POS_MONTHS_BALANCE_MAX | 9704.941992 |
| numeric__BURO_DAYS_CREDIT_MAX | 9603.360861 |
