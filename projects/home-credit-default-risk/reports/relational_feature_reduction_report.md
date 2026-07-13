# Home Credit Relational Feature Reduction Report

Run date: 2026-07-13

## Why The Feature Count Grew

The feature count grew from about 100 application features to 908 joined raw features because each relational table has many rows per applicant. The pipeline first aggregates those histories to `SK_ID_CURR`; each numeric field creates several statistics such as mean, max, min, and sum, and each categorical field creates count/rate columns for its observed values. `previous_application` is the largest source because it has many product, status, purpose, channel, and category fields.

| feature_group | raw_source_columns | positive_gain_columns | zero_gain_columns | total_gain |
| --- | --- | --- | --- | --- |
| previous_application | 407 | 246 | 161 | 315881.972656 |
| bureau + bureau_balance | 174 | 110 | 64 | 247875.247353 |
| application engineered | 128 | 121 | 7 | 1493736.352121 |
| credit_card_balance | 109 | 81 | 28 | 89142.747593 |
| installments_payments | 45 | 39 | 6 | 191982.270847 |
| POS_CASH_balance | 45 | 31 | 14 | 64542.929884 |

## Profile JSON Review

The profile JSONs confirm which fields are structurally risky: high-missing fields are useful only when aggregated carefully, and high-cardinality IDs must never be used directly. The modeling pipeline uses `SK_ID_CURR` only for joins and does not train on raw ID columns.

| table | feature | type | p_missing | p_distinct | n_distinct |
| --- | --- | --- | --- | --- | --- |
| previous_application | RATE_INTEREST_PRIMARY | Numeric | 0.996450 | 0.067606 | 48 |
| previous_application | RATE_INTEREST_PRIVILEGED | Numeric | 0.996450 | 0.016901 | 12 |
| bureau | AMT_ANNUITY | Numeric | 0.719690 | 0.193304 | 10837 |
| application_test | COMMONAREA_AVG | Numeric | 0.687161 | 0.133910 | 2042 |
| application_test | COMMONAREA_MEDI | Numeric | 0.687161 | 0.133386 | 2034 |
| application_test | COMMONAREA_MODE | Numeric | 0.687161 | 0.131222 | 2001 |
| application_test | NONLIVINGAPARTMENTS_AVG | Numeric | 0.684125 | 0.015652 | 241 |
| application_test | NONLIVINGAPARTMENTS_MEDI | Numeric | 0.684125 | 0.008703 | 134 |
| application_test | NONLIVINGAPARTMENTS_MODE | Numeric | 0.684125 | 0.006884 | 106 |
| application_test | FONDKAPREMONT_MODE | Text | 0.672842 | 0.000251 | 4 |
| application_test | LIVINGAPARTMENTS_AVG | Numeric | 0.672493 | 0.075858 | 1211 |
| application_test | LIVINGAPARTMENTS_MEDI | Numeric | 0.672493 | 0.052806 | 843 |
| application_test | LIVINGAPARTMENTS_MODE | Numeric | 0.672493 | 0.037710 | 602 |
| application_test | FLOORSMIN_AVG | Numeric | 0.666051 | 0.012164 | 198 |
| application_test | FLOORSMIN_MEDI | Numeric | 0.666051 | 0.002703 | 44 |
| application_test | FLOORSMIN_MODE | Numeric | 0.666051 | 0.001536 | 25 |
| application_test | OWN_CAR_AGE | Numeric | 0.662892 | 0.003165 | 52 |
| bureau | AMT_CREDIT_MAX_OVERDUE | Numeric | 0.658745 | 0.160554 | 10958 |
| application_test | YEARS_BUILD_MODE | Numeric | 0.652757 | 0.007799 | 132 |
| application_test | YEARS_BUILD_AVG | Numeric | 0.652757 | 0.007680 | 130 |
| application_test | YEARS_BUILD_MEDI | Numeric | 0.652757 | 0.007621 | 129 |
| application_test | LANDAREA_MEDI | Numeric | 0.579641 | 0.125037 | 2562 |
| application_test | LANDAREA_MODE | Numeric | 0.579641 | 0.124939 | 2560 |
| application_test | LANDAREA_AVG | Numeric | 0.579641 | 0.123963 | 2540 |
| application_test | BASEMENTAREA_MODE | Numeric | 0.567065 | 0.134341 | 2835 |
| application_test | BASEMENTAREA_AVG | Numeric | 0.567065 | 0.133441 | 2816 |
| application_test | BASEMENTAREA_MEDI | Numeric | 0.567065 | 0.132919 | 2805 |
| application_test | NONLIVINGAREA_MEDI | Numeric | 0.535122 | 0.089585 | 2030 |
| application_test | NONLIVINGAREA_AVG | Numeric | 0.535122 | 0.089409 | 2026 |
| application_test | NONLIVINGAREA_MODE | Numeric | 0.535122 | 0.089365 | 2025 |
| application_test | ELEVATORS_AVG | Numeric | 0.516761 | 0.007684 | 181 |
| application_test | ELEVATORS_MEDI | Numeric | 0.516761 | 0.001826 | 43 |
| application_test | ELEVATORS_MODE | Numeric | 0.516761 | 0.001104 | 26 |
| previous_application | RATE_DOWN_PAYMENT | Numeric | 0.512485 | 0.382624 | 37307 |
| previous_application | AMT_DOWN_PAYMENT | Numeric | 0.512485 | 0.094889 | 9252 |
| application_test | WALLSMATERIAL_MODE | Text | 0.490173 | 0.000282 | 7 |
| application_test | APARTMENTS_AVG | Numeric | 0.490050 | 0.062075 | 1543 |
| application_test | APARTMENTS_MEDI | Numeric | 0.490050 | 0.036931 | 918 |
| application_test | APARTMENTS_MODE | Numeric | 0.490050 | 0.025586 | 636 |
| previous_application | NAME_TYPE_SUITE | Text | 0.486655 | 0.000068 | 7 |

## Reduction Validation Results

| feature_set | raw_feature_count | transformed_feature_count | numeric_add_indicator | one_hot_min_frequency | one_hot_max_categories | threshold | validation_roc_auc | validation_average_precision | validation_accuracy | validation_precision_class_1 | validation_recall_class_1 | validation_f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_200_no_missing_indicators | 200 | 244 | False | 1000 | 15 | 0.673881 | 0.787824 | 0.272900 | 0.857546 | 0.268062 | 0.441843 | 0.333682 |
| top_150_no_missing_indicators | 150 | 194 | False | 1000 | 15 | 0.685435 | 0.787486 | 0.271720 | 0.863319 | 0.274307 | 0.421198 | 0.332241 |
| top_250_no_missing_indicators | 250 | 294 | False | 1000 | 15 | 0.691763 | 0.787466 | 0.273549 | 0.868135 | 0.282879 | 0.412638 | 0.335654 |
| domain_250_no_missing_indicators | 250 | 280 | False | 1000 | 15 | 0.671277 | 0.786004 | 0.271589 | 0.857526 | 0.270058 | 0.449144 | 0.337304 |
| domain_200_no_missing_indicators | 200 | 200 | False | 1000 | 15 | 0.656042 | 0.780555 | 0.265048 | 0.846835 | 0.253799 | 0.462487 | 0.327743 |
| domain_150_no_missing_indicators | 150 | 150 | False | 1000 | 15 | 0.676129 | 0.780197 | 0.265583 | 0.857668 | 0.263536 | 0.425227 | 0.325402 |

## Final Holdout Metrics

| feature_set | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_200_no_missing_indicators | default_0.5 | 0.500000 | 0.790318 | 0.287613 | 0.738240 | 0.192329 | 0.700906 | 0.301834 |
| top_200_no_missing_indicators | validation_selected | 0.673881 | 0.790318 | 0.287613 | 0.861617 | 0.278762 | 0.449950 | 0.344248 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,758 | 5,780 |
| Actual 1 | 2,731 | 2,234 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.90      0.92     56538
           1       0.28      0.45      0.34      4965

    accuracy                           0.86     61503
   macro avg       0.61      0.67      0.63     61503
weighted avg       0.89      0.86      0.88     61503

```

## Final Feature Importance

| transformed_feature | gain_importance |
| --- | --- |
| numeric__EXT_SOURCE_MEAN | 772804.626526 |
| numeric__EXT_SOURCE_MIN | 94163.045300 |
| numeric__CREDIT_TERM_APPROX | 70346.849546 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MAX | 57422.853514 |
| numeric__EXT_SOURCE_MAX | 57393.342327 |
| numeric__INST_INST_LATE_PAYMENT_FLAG_MEAN | 48106.255611 |
| numeric__GOODS_CREDIT_RATIO | 44975.756300 |
| numeric__AGE_YEARS | 38309.242212 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_MAX | 35227.015314 |
| numeric__PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN | 30345.119717 |
| numeric__AMT_ANNUITY | 29814.894907 |
| numeric__PREV_NAME_CONTRACT_STATUS_REFUSED_RATE | 23058.053345 |
| numeric__INST_AMT_PAYMENT_SUM | 22458.705816 |
| numeric__CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | 22441.945192 |
| categorical__NAME_EDUCATION_TYPE_Higher education | 22361.015244 |
| numeric__EXT_SOURCE_3 | 18710.042051 |
| numeric__DAYS_EMPLOYED | 18602.240711 |
| numeric__AMT_GOODS_PRICE | 18462.595106 |
| categorical__CODE_GENDER_M | 17608.981874 |
| numeric__EMPLOYMENT_AGE_RATIO | 17606.829103 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MEAN | 17394.306114 |
| categorical__CODE_GENDER_F | 15778.465179 |
| numeric__EMPLOYED_YEARS | 15396.172071 |
| numeric__ANNUITY_INCOME_RATIO | 14985.335245 |
| numeric__EXT_SOURCE_1 | 14506.197166 |
| numeric__EXT_SOURCE_COUNT | 13681.714998 |
| numeric__AMT_CREDIT | 12332.012999 |
| numeric__CC_CC_UTILIZATION_RATIO_MAX | 12305.836779 |
| numeric__OWN_CAR_AGE | 11821.718533 |
| numeric__INST_AMT_PAYMENT_MEAN | 11568.132469 |
| numeric__EXT_SOURCE_2 | 11525.486092 |
| numeric__POS_POS_COMPLETION_RATIO_SUM | 11286.010025 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_SUM | 10626.768280 |
| numeric__INST_INST_PAYMENT_DIFF_MEAN | 10518.874647 |
| numeric__PREV_DAYS_LAST_DUE_MAX | 10410.049231 |
| numeric__ANNUITY_PER_PERSON | 10008.740999 |
| numeric__BURO_DAYS_CREDIT_ENDDATE_MAX | 9899.551067 |
| numeric__CC_CC_UTILIZATION_RATIO_MEAN | 9498.229813 |
| numeric__POS_POS_COMPLETION_RATIO_MEAN | 9472.274542 |
| numeric__BURO_DAYS_ENDDATE_FACT_MAX | 9366.981794 |
| numeric__CC_CNT_DRAWINGS_CURRENT_MEAN | 8839.444805 |
| numeric__INST_INST_PAYMENT_DELAY_MAX | 8793.739401 |
| numeric__CC_CC_DRAWING_LIMIT_RATIO_MEAN | 8519.863621 |
| numeric__POS_MONTHS_BALANCE_MAX | 8379.654785 |
| numeric__DAYS_ID_PUBLISH | 8365.834797 |
| numeric__PREV_PREV_CREDIT_APPLICATION_RATIO_MAX | 8332.175936 |
| numeric__PREV_NAME_YIELD_GROUP_LOW_ACTION_RATE | 8309.407377 |
| numeric__BURO_AMT_CREDIT_SUM_SUM | 8306.626337 |
| numeric__BURO_AMT_CREDIT_SUM_MAX | 8286.570473 |
| numeric__PREV_CNT_PAYMENT_MEAN | 8207.894238 |

## Model Saving

Saved model: `C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_relational_reduced_lgbm.joblib`

## Recommendation

Use the smallest model whose validation and holdout metrics remain close to the full relational model. Further feature count reduction should be done by changing the relational aggregation recipe itself, especially by removing rare previous-application categorical count/rate features, not by adding more one-hot controls.
