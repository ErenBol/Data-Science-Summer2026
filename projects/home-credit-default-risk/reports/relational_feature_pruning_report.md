# Home Credit Relational Feature Pruning Report

Run date: 2026-07-13

## Purpose

This experiment keeps the same train/validation/test split and the same LightGBM hyperparameters, then tests whether the joined relational model can be made smaller by dropping zero/low-gain raw columns and reducing pipeline expansion.

Controls tested:

- grouped application one-hot encoding with `min_frequency=1000` and `max_categories=20`
- removing numeric imputer missing-indicator expansion
- keeping only raw columns mapped to positive transformed gain
- keeping top raw columns by summed transformed gain
- keeping a domain-compact subset from repayment, bureau, previous-loan, POS, credit-card, and core application groups

## Validation Results

| feature_set | raw_feature_count | transformed_feature_count | numeric_add_indicator | one_hot_min_frequency | one_hot_max_categories | threshold | validation_roc_auc | validation_average_precision | validation_accuracy | validation_precision_class_1 | validation_recall_class_1 | validation_f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| positive_gain_raw_grouped_ohe | 628 | 1284 | True | 1000 | 20 | 0.669277 | 0.787927 | 0.272846 | 0.856916 | 0.267646 | 0.444864 | 0.334216 |
| domain_compact_350_no_missing_indicators | 350 | 399 | False | 1000 | 20 | 0.682093 | 0.787521 | 0.272782 | 0.863380 | 0.276931 | 0.429758 | 0.336819 |
| top_450_raw_no_missing_indicators | 450 | 510 | False | 1000 | 20 | 0.657600 | 0.787442 | 0.273907 | 0.850270 | 0.261285 | 0.467774 | 0.335288 |
| all_features_no_missing_indicators | 908 | 974 | False | 1000 | 20 | 0.644462 | 0.787375 | 0.272529 | 0.842852 | 0.253927 | 0.488419 | 0.334137 |
| positive_gain_raw_no_missing_indicators | 628 | 694 | False | 1000 | 20 | 0.686036 | 0.787305 | 0.273519 | 0.865737 | 0.279545 | 0.420443 | 0.335813 |
| top_300_raw_no_missing_indicators | 300 | 349 | False | 1000 | 20 | 0.692981 | 0.786828 | 0.272682 | 0.868501 | 0.282783 | 0.409366 | 0.334499 |

## Final Holdout Metrics

| feature_set | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| positive_gain_raw_grouped_ohe | default_0.5 | 0.500000 | 0.790043 | 0.288695 | 0.738631 | 0.192039 | 0.697684 | 0.301178 |
| positive_gain_raw_grouped_ohe | validation_selected | 0.669277 | 0.790043 | 0.288695 | 0.859649 | 0.275774 | 0.454179 | 0.343175 |

## Compact Domain Holdout Check

The compact domain variant is shown separately because it gives the best feature-count tradeoff.

| feature_set | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| domain_compact_350_no_missing_indicators | default_0.5 | 0.500000 | 0.790435 | 0.289425 | 0.738354 | 0.191961 | 0.698288 | 0.301138 |
| domain_compact_350_no_missing_indicators | validation_selected | 0.682093 | 0.790435 | 0.289425 | 0.867193 | 0.287289 | 0.435650 | 0.346246 |

Compact domain confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 51,172 | 5,366 |
| Actual 1 | 2,802 | 2,163 |

Compact domain top feature importances:

| transformed_feature | gain_importance |
| --- | --- |
| numeric__EXT_SOURCE_MEAN | 742010.401367 |
| numeric__EXT_SOURCE_MIN | 104920.911734 |
| numeric__CREDIT_TERM_APPROX | 69141.412344 |
| numeric__EXT_SOURCE_MAX | 67467.253616 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MAX | 57211.238388 |
| numeric__INST_INST_LATE_PAYMENT_FLAG_MEAN | 47486.430962 |
| numeric__GOODS_CREDIT_RATIO | 43679.736755 |
| numeric__AGE_YEARS | 36348.240257 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_MAX | 33564.875463 |
| numeric__PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN | 29761.266792 |
| numeric__AMT_ANNUITY | 28861.357054 |
| categorical__NAME_EDUCATION_TYPE_Higher education | 22312.135075 |
| numeric__INST_AMT_PAYMENT_SUM | 22214.465183 |
| numeric__CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | 21037.869900 |
| numeric__PREV_NAME_CONTRACT_STATUS_REFUSED_RATE | 20792.979591 |
| numeric__EXT_SOURCE_3 | 20011.498350 |
| categorical__CODE_GENDER_M | 17932.661339 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MEAN | 17647.035065 |
| numeric__DAYS_EMPLOYED | 16947.407373 |
| numeric__EMPLOYMENT_AGE_RATIO | 16864.788961 |
| numeric__AMT_GOODS_PRICE | 16697.237415 |
| numeric__EMPLOYED_YEARS | 16192.298166 |
| numeric__EXT_SOURCE_1 | 15254.400795 |
| categorical__CODE_GENDER_F | 14976.303131 |
| numeric__EXT_SOURCE_COUNT | 14704.686367 |
| numeric__ANNUITY_INCOME_RATIO | 14103.321096 |
| numeric__EXT_SOURCE_2 | 13585.013502 |
| numeric__CC_CC_UTILIZATION_RATIO_MAX | 12231.621784 |
| numeric__AMT_CREDIT | 11634.178253 |
| numeric__OWN_CAR_AGE | 11524.269630 |
| numeric__POS_POS_COMPLETION_RATIO_SUM | 10737.000135 |
| numeric__INST_AMT_PAYMENT_MEAN | 10633.313549 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_SUM | 10301.899982 |
| numeric__PREV_DAYS_LAST_DUE_MAX | 9984.905813 |
| numeric__ANNUITY_PER_PERSON | 9531.612812 |
| numeric__BURO_DAYS_ENDDATE_FACT_MAX | 9429.996683 |
| numeric__BURO_DAYS_CREDIT_ENDDATE_MAX | 9110.917336 |
| numeric__EXT_SOURCE_STD | 9098.981819 |
| numeric__CC_CNT_DRAWINGS_CURRENT_MEAN | 8802.293213 |
| numeric__CREDIT_PER_PERSON | 8795.094431 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,616 | 5,922 |
| Actual 1 | 2,710 | 2,255 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.90      0.92     56538
           1       0.28      0.45      0.34      4965

    accuracy                           0.86     61503
   macro avg       0.61      0.67      0.63     61503
weighted avg       0.89      0.86      0.87     61503

```

## Top Final Feature Importances

| transformed_feature | gain_importance |
| --- | --- |
| numeric__EXT_SOURCE_MEAN | 784736.508459 |
| numeric__EXT_SOURCE_MIN | 89092.321268 |
| numeric__CREDIT_TERM_APPROX | 66308.989763 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MAX | 55405.263220 |
| numeric__EXT_SOURCE_MAX | 49784.933077 |
| numeric__INST_INST_LATE_PAYMENT_FLAG_MEAN | 46739.496735 |
| numeric__GOODS_CREDIT_RATIO | 44598.404730 |
| numeric__AGE_YEARS | 37021.681774 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_MAX | 33583.764294 |
| numeric__AMT_ANNUITY | 29075.401876 |
| numeric__PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN | 28725.616329 |
| numeric__INST_AMT_PAYMENT_SUM | 23581.083117 |
| categorical__NAME_EDUCATION_TYPE_Higher education | 21758.807281 |
| numeric__CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | 21076.972103 |
| numeric__PREV_NAME_CONTRACT_STATUS_REFUSED_RATE | 20426.334385 |
| numeric__DAYS_EMPLOYED | 19747.165003 |
| numeric__AMT_GOODS_PRICE | 17823.030706 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MEAN | 17334.788040 |
| categorical__CODE_GENDER_F | 16315.184155 |
| categorical__CODE_GENDER_M | 15922.623299 |
| numeric__EMPLOYMENT_AGE_RATIO | 15762.054403 |
| numeric__ANNUITY_INCOME_RATIO | 15262.507595 |
| numeric__EXT_SOURCE_3 | 14975.986694 |
| numeric__missingindicator_EXT_SOURCE_1 | 14787.136587 |
| numeric__EMPLOYED_YEARS | 13342.105568 |
| numeric__CC_CC_UTILIZATION_RATIO_MAX | 11096.092823 |
| numeric__AMT_CREDIT | 10831.077871 |
| numeric__INST_AMT_PAYMENT_MEAN | 10633.737309 |
| numeric__OWN_CAR_AGE | 10557.819824 |
| numeric__POS_POS_COMPLETION_RATIO_SUM | 10343.891514 |
| numeric__EXT_SOURCE_2 | 10246.976006 |
| numeric__INST_INST_PAYMENT_DIFF_MEAN | 9808.225222 |
| numeric__BURO_DAYS_CREDIT_ENDDATE_MAX | 9123.156582 |
| numeric__INST_AMT_PAYMENT_MIN | 8911.154900 |
| numeric__PREV_DAYS_LAST_DUE_MAX | 8805.525225 |
| numeric__CC_CC_UTILIZATION_RATIO_MEAN | 8734.863987 |
| categorical__NAME_FAMILY_STATUS_Married | 8697.289988 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_SUM | 8487.167946 |
| numeric__ANNUITY_PER_PERSON | 8419.294289 |
| numeric__PREV_NAME_YIELD_GROUP_LOW_ACTION_RATE | 8122.534115 |

## Raw Column Gain Summary

| source_column | total_gain | max_gain | transformed_feature_count | positive_transformed_feature_count | source_column_exists |
| --- | --- | --- | --- | --- | --- |
| EXT_SOURCE_MEAN | 754268.248363 | 754268.248363 | 2 | 1 | True |
| EXT_SOURCE_MIN | 111120.871225 | 111120.871225 | 2 | 1 | True |
| CREDIT_TERM_APPROX | 67951.873411 | 67951.873411 | 2 | 1 | True |
| EXT_SOURCE_MAX | 59565.386303 | 59565.386303 | 2 | 1 | True |
| BURO_BURO_DEBT_CREDIT_RATIO_MAX | 56662.530704 | 56662.530704 | 2 | 1 | True |
| INST_INST_LATE_PAYMENT_FLAG_MEAN | 49238.302052 | 49238.302052 | 2 | 1 | True |
| GOODS_CREDIT_RATIO | 44594.020910 | 44594.020910 | 2 | 1 | True |
| AGE_YEARS | 36856.707182 | 36856.707182 | 1 | 1 | True |
| PREV_DAYS_LAST_DUE_1ST_VERSION_MAX | 32552.885391 | 32552.885391 | 2 | 1 | True |
| CODE_GENDER | 32340.690775 | 17534.539440 | 3 | 2 | True |
| PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN | 29993.906042 | 29993.906042 | 2 | 1 | True |
| NAME_EDUCATION_TYPE | 27914.072855 | 22083.057976 | 5 | 4 | True |
| AMT_ANNUITY | 27542.803493 | 27542.803493 | 2 | 1 | True |
| EXT_SOURCE_1 | 21264.481417 | 14644.150005 | 2 | 2 | True |
| CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | 20476.515430 | 20476.515430 | 2 | 1 | True |
| INST_AMT_PAYMENT_SUM | 20448.261246 | 20448.261246 | 2 | 1 | True |
| PREV_NAME_CONTRACT_STATUS_REFUSED_RATE | 20394.188406 | 20394.188406 | 2 | 1 | True |
| DAYS_EMPLOYED | 17668.808262 | 17625.092262 | 2 | 2 | True |
| AMT_GOODS_PRICE | 16866.166920 | 16866.166920 | 2 | 1 | True |
| BURO_BURO_DEBT_CREDIT_RATIO_MEAN | 16734.352825 | 16008.692822 | 2 | 2 | True |
| EXT_SOURCE_3 | 15660.591253 | 13876.559649 | 2 | 2 | True |
| EMPLOYMENT_AGE_RATIO | 15446.989801 | 15446.989801 | 2 | 1 | True |
| EMPLOYED_YEARS | 15375.663853 | 15375.663853 | 2 | 1 | True |
| OWN_CAR_AGE | 14791.527107 | 11670.912323 | 2 | 2 | True |
| ANNUITY_INCOME_RATIO | 14093.593439 | 14093.593439 | 2 | 1 | True |
| AMT_CREDIT | 12395.016438 | 12395.016438 | 1 | 1 | True |
| POS_POS_COMPLETION_RATIO_SUM | 11984.512547 | 11984.512547 | 2 | 1 | True |
| ORGANIZATION_TYPE | 11153.134106 | 2093.715004 | 58 | 16 | True |
| INST_AMT_PAYMENT_MEAN | 11102.825933 | 11102.825933 | 2 | 1 | True |
| CC_CC_UTILIZATION_RATIO_MAX | 10588.035496 | 10588.035496 | 2 | 1 | True |
| EXT_SOURCE_2 | 10144.705025 | 10144.705025 | 2 | 1 | True |
| ANNUITY_PER_PERSON | 10098.991499 | 10098.991499 | 2 | 1 | True |
| BURO_DAYS_CREDIT_ENDDATE_MAX | 9930.790617 | 9930.790617 | 2 | 1 | True |
| PREV_DAYS_LAST_DUE_MAX | 9902.459387 | 9902.459387 | 2 | 1 | True |
| PREV_DAYS_LAST_DUE_1ST_VERSION_SUM | 9108.509422 | 9108.509422 | 2 | 1 | True |
| INST_INST_PAYMENT_DIFF_MEAN | 8996.830288 | 8996.830288 | 2 | 1 | True |
| NAME_FAMILY_STATUS | 8585.345877 | 8328.552076 | 6 | 3 | True |
| INST_AMT_PAYMENT_MIN | 8584.003796 | 8584.003796 | 2 | 1 | True |
| CC_CNT_DRAWINGS_CURRENT_MEAN | 8347.631248 | 8347.631248 | 2 | 1 | True |
| NAME_INCOME_TYPE | 8323.270725 | 5530.558712 | 8 | 4 | True |

## Findings

- `create_features()` is appropriate for the application table only. It should not be expanded to relational tables; the relational tables need grouped aggregations before joining.
- The >1000 transformed-feature count mostly came from relational category count/rate aggregates plus numeric missing indicators, not just application one-hot encoding.
- One-hot grouping helps control rare application categories, but it is not enough by itself because application categoricals are a small share of the transformed matrix.
- Gain-based raw-column pruning is the direct way to reduce the joined feature matrix while preserving the same modeling setup.
