# Home Credit Relational Feature Experiments Report

Run date: 2026-07-13

## Setup

- Base model: notebook 09 expanded application feature set.
- Model: LightGBM with the same hyperparameters as the last selected setup.
- Split: same random state, 20% holdout test, then 20% validation from the training split.
- Threshold rule: same validation max-F1 threshold selection used in the previous notebooks.
- Relational tables were aggregated to one row per `SK_ID_CURR` before joining.

## Validation Results

| feature_set | raw_feature_count | threshold | validation_roc_auc | validation_average_precision | validation_accuracy | validation_precision_class_1 | validation_recall_class_1 | validation_f1_class_1 | groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_plus_bureau_plus_previous_application_plus_installments_payments_plus_POS_CASH_balance_plus_credit_card_balance | 908 | 0.687520 | 0.787245 | 0.272062 | 0.866814 | 0.280788 | 0.416163 | 0.335328 | bureau,previous_application,installments_payments,POS_CASH_balance,credit_card_balance |
| app_plus_bureau_plus_previous_application_plus_installments_payments | 754 | 0.670884 | 0.786036 | 0.269483 | 0.857424 | 0.267887 | 0.442095 | 0.333618 | bureau,previous_application,installments_payments |
| app_plus_bureau_plus_previous_application_plus_installments_payments_plus_POS_CASH_balance | 799 | 0.677015 | 0.785561 | 0.271133 | 0.860392 | 0.270335 | 0.429255 | 0.331744 | bureau,previous_application,installments_payments,POS_CASH_balance |
| app_plus_bureau_plus_previous_application | 709 | 0.673541 | 0.779348 | 0.260155 | 0.856835 | 0.263765 | 0.431772 | 0.327478 | bureau,previous_application |
| app_plus_installments_payments | 173 | 0.639584 | 0.773751 | 0.255888 | 0.834600 | 0.237591 | 0.474824 | 0.316709 | installments_payments |
| app_plus_previous_application | 535 | 0.665006 | 0.772954 | 0.250844 | 0.850026 | 0.251930 | 0.435549 | 0.319218 | previous_application |
| app_plus_bureau | 302 | 0.668245 | 0.770063 | 0.256343 | 0.852486 | 0.253525 | 0.425478 | 0.317729 | bureau |
| app_plus_POS_CASH_balance | 173 | 0.670222 | 0.768145 | 0.252689 | 0.851957 | 0.252762 | 0.426234 | 0.317338 | POS_CASH_balance |
| app_plus_credit_card_balance | 237 | 0.660810 | 0.767240 | 0.248766 | 0.845047 | 0.243755 | 0.437311 | 0.313029 | credit_card_balance |
| app_expanded_only | 128 | 0.688844 | 0.762402 | 0.243919 | 0.861225 | 0.257637 | 0.382175 | 0.307786 | nan |

## Selected Configuration

Best validation ROC-AUC feature set: `app_plus_bureau_plus_previous_application_plus_installments_payments_plus_POS_CASH_balance_plus_credit_card_balance`

Included relational groups: `bureau, previous_application, installments_payments, POS_CASH_balance, credit_card_balance`

## Holdout Test Metrics

| feature_set | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| app_plus_bureau_plus_previous_application_plus_installments_payments_plus_POS_CASH_balance_plus_credit_card_balance | default_0.5 | 0.500000 | 0.790203 | 0.288818 | 0.738956 | 0.192184 | 0.697281 | 0.301319 |
| app_plus_bureau_plus_previous_application_plus_installments_payments_plus_POS_CASH_balance_plus_credit_card_balance | validation_selected | 0.687520 | 0.790203 | 0.288818 | 0.869600 | 0.289514 | 0.423162 | 0.343806 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 51,382 | 5,156 |
| Actual 1 | 2,864 | 2,101 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.91      0.93     56538
           1       0.29      0.42      0.34      4965

    accuracy                           0.87     61503
   macro avg       0.62      0.67      0.64     61503
weighted avg       0.89      0.87      0.88     61503

```

## Top Final Feature Importances

| transformed_feature | gain_importance |
| --- | --- |
| numeric__EXT_SOURCE_MEAN | 754268.248363 |
| numeric__EXT_SOURCE_MIN | 111120.871225 |
| numeric__CREDIT_TERM_APPROX | 67951.873411 |
| numeric__EXT_SOURCE_MAX | 59565.386303 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MAX | 56662.530704 |
| numeric__INST_INST_LATE_PAYMENT_FLAG_MEAN | 49238.302052 |
| numeric__GOODS_CREDIT_RATIO | 44594.020910 |
| numeric__AGE_YEARS | 36856.707182 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_MAX | 32552.885391 |
| numeric__PREV_PREV_CREDIT_APPLICATION_RATIO_MEAN | 29993.906042 |
| numeric__AMT_ANNUITY | 27542.803493 |
| categorical__NAME_EDUCATION_TYPE_Higher education | 22083.057976 |
| numeric__CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | 20476.515430 |
| numeric__INST_AMT_PAYMENT_SUM | 20448.261246 |
| numeric__PREV_NAME_CONTRACT_STATUS_REFUSED_RATE | 20394.188406 |
| numeric__DAYS_EMPLOYED | 17625.092262 |
| categorical__CODE_GENDER_M | 17534.539440 |
| numeric__AMT_GOODS_PRICE | 16866.166920 |
| numeric__BURO_BURO_DEBT_CREDIT_RATIO_MEAN | 16008.692822 |
| numeric__EMPLOYMENT_AGE_RATIO | 15446.989801 |
| numeric__EMPLOYED_YEARS | 15375.663853 |
| categorical__CODE_GENDER_F | 14806.151335 |
| numeric__missingindicator_EXT_SOURCE_1 | 14644.150005 |
| numeric__ANNUITY_INCOME_RATIO | 14093.593439 |
| numeric__EXT_SOURCE_3 | 13876.559649 |
| numeric__AMT_CREDIT | 12395.016438 |
| numeric__POS_POS_COMPLETION_RATIO_SUM | 11984.512547 |
| numeric__OWN_CAR_AGE | 11670.912323 |
| numeric__INST_AMT_PAYMENT_MEAN | 11102.825933 |
| numeric__CC_CC_UTILIZATION_RATIO_MAX | 10588.035496 |
| numeric__EXT_SOURCE_2 | 10144.705025 |
| numeric__ANNUITY_PER_PERSON | 10098.991499 |
| numeric__BURO_DAYS_CREDIT_ENDDATE_MAX | 9930.790617 |
| numeric__PREV_DAYS_LAST_DUE_MAX | 9902.459387 |
| numeric__PREV_DAYS_LAST_DUE_1ST_VERSION_SUM | 9108.509422 |
| numeric__INST_INST_PAYMENT_DIFF_MEAN | 8996.830288 |
| numeric__INST_AMT_PAYMENT_MIN | 8584.003796 |
| numeric__CC_CNT_DRAWINGS_CURRENT_MEAN | 8347.631248 |
| categorical__NAME_FAMILY_STATUS_Married | 8328.552076 |
| numeric__CC_CC_UTILIZATION_RATIO_MEAN | 8312.288914 |

## Interpretation

The final model still uses the application-table engineered signals, but it now also learns from historical credit behavior summarized across the secondary Home Credit tables. The most important relational signals should be interpreted as customer-level aggregates, not individual loan records.

If the result remains below 0.80 ROC-AUC, the next performance step is not more threshold work. Thresholds change precision/recall tradeoffs, not ranking quality. The next step is LightGBM tuning on the joined relational feature matrix, plus more targeted aggregate features around recent history windows and category-specific bureau/previous-loan behavior.
