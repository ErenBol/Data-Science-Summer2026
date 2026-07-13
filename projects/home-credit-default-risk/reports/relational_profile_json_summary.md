# Relational Profile JSON Summary

## application_test

Highest missingness fields:
- `COMMONAREA_AVG`: missing 0.687, type Numeric
- `COMMONAREA_MEDI`: missing 0.687, type Numeric
- `COMMONAREA_MODE`: missing 0.687, type Numeric
- `NONLIVINGAPARTMENTS_AVG`: missing 0.684, type Numeric
- `NONLIVINGAPARTMENTS_MEDI`: missing 0.684, type Numeric
- `NONLIVINGAPARTMENTS_MODE`: missing 0.684, type Numeric
- `FONDKAPREMONT_MODE`: missing 0.673, type Text
- `LIVINGAPARTMENTS_MEDI`: missing 0.672, type Numeric

Highest distinct-rate fields:
- `SK_ID_CURR`: distinct-rate 1.000, n_distinct 48744
- `EXT_SOURCE_1`: distinct-rate 0.964, n_distinct 27207
- `EXT_SOURCE_2`: distinct-rate 0.798, n_distinct 38885
- `DAYS_BIRTH`: distinct-rate 0.318, n_distinct 15477
- `DAYS_REGISTRATION`: distinct-rate 0.259, n_distinct 12618
- `DAYS_EMPLOYED`: distinct-rate 0.161, n_distinct 7863
- `LIVINGAREA_MEDI`: distinct-rate 0.154, n_distinct 3885
- `AMT_ANNUITY`: distinct-rate 0.154, n_distinct 7491

## application_train

Highest missingness fields:
- `COMMONAREA_AVG`: missing 0.699, type Numeric
- `COMMONAREA_MODE`: missing 0.699, type Numeric
- `COMMONAREA_MEDI`: missing 0.699, type Numeric
- `NONLIVINGAPARTMENTS_MEDI`: missing 0.694, type Numeric
- `NONLIVINGAPARTMENTS_MODE`: missing 0.694, type Numeric
- `NONLIVINGAPARTMENTS_AVG`: missing 0.694, type Numeric
- `FONDKAPREMONT_MODE`: missing 0.684, type Text
- `LIVINGAPARTMENTS_AVG`: missing 0.684, type Numeric

Highest distinct-rate fields:
- `SK_ID_CURR`: distinct-rate 1.000, n_distinct 307511
- `EXT_SOURCE_1`: distinct-rate 0.854, n_distinct 114584
- `EXT_SOURCE_2`: distinct-rate 0.391, n_distinct 119831
- `DAYS_BIRTH`: distinct-rate 0.057, n_distinct 17460
- `DAYS_REGISTRATION`: distinct-rate 0.051, n_distinct 15688
- `AMT_ANNUITY`: distinct-rate 0.044, n_distinct 13672
- `DAYS_EMPLOYED`: distinct-rate 0.041, n_distinct 12574
- `LIVINGAREA_MODE`: distinct-rate 0.035, n_distinct 5301

## bureau_balance

Highest missingness fields:
- `SK_ID_BUREAU`: missing 0.000, type Numeric
- `MONTHS_BALANCE`: missing 0.000, type Numeric
- `STATUS`: missing 0.000, type Text

Highest distinct-rate fields:
- `SK_ID_BUREAU`: distinct-rate 0.031, n_distinct 6119
- `MONTHS_BALANCE`: distinct-rate 0.000, n_distinct 97
- `STATUS`: distinct-rate 0.000, n_distinct 8

## bureau

Highest missingness fields:
- `AMT_ANNUITY`: missing 0.720, type Numeric
- `AMT_CREDIT_MAX_OVERDUE`: missing 0.659, type Numeric
- `DAYS_ENDDATE_FACT`: missing 0.374, type Numeric
- `AMT_CREDIT_SUM_LIMIT`: missing 0.351, type Numeric
- `AMT_CREDIT_SUM_DEBT`: missing 0.150, type Numeric
- `DAYS_CREDIT_ENDDATE`: missing 0.063, type Numeric
- `SK_ID_CURR`: missing 0.000, type Numeric
- `DAYS_CREDIT`: missing 0.000, type Numeric

Highest distinct-rate fields:
- `SK_ID_BUREAU`: distinct-rate 1.000, n_distinct 200000
- `AMT_CREDIT_SUM`: distinct-rate 0.276, n_distinct 55108
- `AMT_CREDIT_SUM_DEBT`: distinct-rate 0.263, n_distinct 44792
- `SK_ID_CURR`: distinct-rate 0.238, n_distinct 47561
- `AMT_ANNUITY`: distinct-rate 0.193, n_distinct 10837
- `AMT_CREDIT_MAX_OVERDUE`: distinct-rate 0.161, n_distinct 10958
- `AMT_CREDIT_SUM_LIMIT`: distinct-rate 0.052, n_distinct 6769
- `DAYS_CREDIT_ENDDATE`: distinct-rate 0.048, n_distinct 8927

## credit_card_balance

Highest missingness fields:
- `AMT_PAYMENT_CURRENT`: missing 0.227, type Numeric
- `CNT_DRAWINGS_POS_CURRENT`: missing 0.226, type Numeric
- `AMT_DRAWINGS_ATM_CURRENT`: missing 0.226, type Numeric
- `CNT_DRAWINGS_ATM_CURRENT`: missing 0.226, type Numeric
- `AMT_DRAWINGS_POS_CURRENT`: missing 0.226, type Numeric
- `AMT_DRAWINGS_OTHER_CURRENT`: missing 0.226, type Numeric
- `CNT_DRAWINGS_OTHER_CURRENT`: missing 0.226, type Numeric
- `CNT_INSTALMENT_MATURE_CUM`: missing 0.059, type Numeric

Highest distinct-rate fields:
- `AMT_TOTAL_RECEIVABLE`: distinct-rate 0.378, n_distinct 75528
- `AMT_RECIVABLE`: distinct-rate 0.378, n_distinct 75527
- `AMT_BALANCE`: distinct-rate 0.368, n_distinct 73536
- `SK_ID_PREV`: distinct-rate 0.354, n_distinct 70809
- `SK_ID_CURR`: distinct-rate 0.352, n_distinct 70494
- `AMT_RECEIVABLE_PRINCIPAL`: distinct-rate 0.339, n_distinct 67751
- `AMT_PAYMENT_CURRENT`: distinct-rate 0.154, n_distinct 23725
- `AMT_INST_MIN_REGULARITY`: distinct-rate 0.118, n_distinct 22116

## installments_payments

Highest missingness fields:
- `SK_ID_PREV`: missing 0.000, type Numeric
- `SK_ID_CURR`: missing 0.000, type Numeric
- `NUM_INSTALMENT_VERSION`: missing 0.000, type Numeric
- `NUM_INSTALMENT_NUMBER`: missing 0.000, type Numeric
- `DAYS_INSTALMENT`: missing 0.000, type Numeric
- `DAYS_ENTRY_PAYMENT`: missing 0.000, type Numeric
- `AMT_INSTALMENT`: missing 0.000, type Numeric
- `AMT_PAYMENT`: missing 0.000, type Numeric

Highest distinct-rate fields:
- `SK_ID_PREV`: distinct-rate 0.585, n_distinct 117000
- `AMT_PAYMENT`: distinct-rate 0.507, n_distinct 101456
- `AMT_INSTALMENT`: distinct-rate 0.488, n_distinct 97596
- `SK_ID_CURR`: distinct-rate 0.329, n_distinct 65867
- `DAYS_ENTRY_PAYMENT`: distinct-rate 0.015, n_distinct 2958
- `DAYS_INSTALMENT`: distinct-rate 0.015, n_distinct 2921
- `NUM_INSTALMENT_NUMBER`: distinct-rate 0.001, n_distinct 221
- `NUM_INSTALMENT_VERSION`: distinct-rate 0.000, n_distinct 35

## POS_CASH_balance

Highest missingness fields:
- `CNT_INSTALMENT`: missing 0.002, type Numeric
- `CNT_INSTALMENT_FUTURE`: missing 0.002, type Numeric
- `SK_ID_CURR`: missing 0.000, type Numeric
- `SK_ID_PREV`: missing 0.000, type Numeric
- `MONTHS_BALANCE`: missing 0.000, type Numeric
- `NAME_CONTRACT_STATUS`: missing 0.000, type Text
- `SK_DPD`: missing 0.000, type Numeric
- `SK_DPD_DEF`: missing 0.000, type Numeric

Highest distinct-rate fields:
- `SK_ID_PREV`: distinct-rate 0.820, n_distinct 164062
- `SK_ID_CURR`: distinct-rate 0.633, n_distinct 126602
- `SK_DPD`: distinct-rate 0.001, n_distinct 155
- `MONTHS_BALANCE`: distinct-rate 0.000, n_distinct 96
- `CNT_INSTALMENT_FUTURE`: distinct-rate 0.000, n_distinct 68
- `CNT_INSTALMENT`: distinct-rate 0.000, n_distinct 56
- `SK_DPD_DEF`: distinct-rate 0.000, n_distinct 51
- `NAME_CONTRACT_STATUS`: distinct-rate 0.000, n_distinct 6

## previous_application

Highest missingness fields:
- `RATE_INTEREST_PRIVILEGED`: missing 0.996, type Numeric
- `RATE_INTEREST_PRIMARY`: missing 0.996, type Numeric
- `AMT_DOWN_PAYMENT`: missing 0.512, type Numeric
- `RATE_DOWN_PAYMENT`: missing 0.512, type Numeric
- `NAME_TYPE_SUITE`: missing 0.487, type Text
- `DAYS_TERMINATION`: missing 0.387, type Numeric
- `DAYS_FIRST_DRAWING`: missing 0.387, type Numeric
- `DAYS_FIRST_DUE`: missing 0.387, type Numeric

Highest distinct-rate fields:
- `SK_ID_PREV`: distinct-rate 1.000, n_distinct 200000
- `SK_ID_CURR`: distinct-rate 0.685, n_distinct 136998
- `AMT_ANNUITY`: distinct-rate 0.585, n_distinct 92015
- `RATE_DOWN_PAYMENT`: distinct-rate 0.383, n_distinct 37307
- `AMT_GOODS_PRICE`: distinct-rate 0.212, n_distinct 33039
- `AMT_CREDIT`: distinct-rate 0.206, n_distinct 41128
- `AMT_APPLICATION`: distinct-rate 0.165, n_distinct 33039
- `AMT_DOWN_PAYMENT`: distinct-rate 0.095, n_distinct 9252

## Feature Hypotheses From Profile JSON

- Bureau tables should add useful credit-history quantity, recency, active/closed status, overdue, debt-to-credit, and bureau-balance delinquency-status signals.
- Previous applications should add approval/refusal history, previous amount ratios, product/channel mix, and decision-recency signals.
- Installments should add direct repayment behavior: late-payment rate, underpayment rate, payment delay, and payment-to-installment ratios.
- POS cash should add active/completed contract status, remaining installments, delinquency days past due, and completion progress.
- Credit card balance should add utilization, payment-to-minimum, drawing behavior, receivables, and DPD signals.
