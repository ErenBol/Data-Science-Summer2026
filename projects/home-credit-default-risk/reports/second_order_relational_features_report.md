# Second-Order Relational Features Report

Run date: 2026-07-17

## Setup

- Added compact second-order recent-history features with `SO_` prefixes in `src/relational_features.py`.
- Appended those features to the current advanced recent/domain block on top of the top-200 reduced base features.
- Kept the same fold-safe target encoding and transformed-feature pruning approach.
- LightGBM CV uses the same 3-fold stratified protocol as `run_advanced_relational_cv.py`.
- Baseline: `advanced_drop_2_weakest_transformed`, CV ROC-AUC `0.792549`, AP `0.288067`.

## Added Second-Order Features

| feature |
| --- |
| SO_BURO_OVERDUE_SUM_6M_TO_24M_RATIO |
| SO_CC_UTILIZATION_CURRENT6_MINUS_PRIOR6 |
| SO_INST_LATE_RATE_3M_MINUS_6M |
| SO_INST_LATE_RATE_6M_MINUS_12M |
| SO_INST_LATE_RATE_ACCEL_3_6_12M |
| SO_PREV_RECENCY_WEIGHTED_REFUSAL_RATE |
| SO_POS_DPD_RATE_3M_MINUS_6M |
| SO_POS_DPD_RATE_6M_MINUS_12M |
| SO_POS_DPD_RATE_ACCEL_3_6_12M |
| SO_POS_DPD_DEF_RATE_3M_MINUS_6M |
| SO_POS_DPD_DEF_RATE_6M_MINUS_12M |

## LightGBM CV Results

| feature_set | fold | raw_feature_count | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| advanced_drop_2_plus_second_order | 1 | 332 | 374 | 0.792758 | 0.292341 |
| advanced_drop_2_plus_second_order | 2 | 332 | 374 | 0.792468 | 0.286737 |
| advanced_drop_2_plus_second_order | 3 | 332 | 374 | 0.792330 | 0.286309 |

## CV Summary

| feature_set | mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision | mean_transformed_feature_count |
| --- | --- | --- | --- | --- | --- |
| advanced_drop_2_weakest_transformed_baseline | 0.792549 | nan | 0.288067 | nan | 363.000000 |
| advanced_drop_2_plus_second_order | 0.792518 | 0.000178 | 0.288462 | 0.002748 | 374.000000 |

Delta versus baseline:

| metric | delta |
| --- | --- |
| mean_cv_roc_auc | -0.000030 |
| mean_cv_average_precision | 0.000395 |

The second-order LightGBM CV ROC-AUC did not improve over the baseline, so the 60/40 LightGBM+CatBoost holdout blend was not retrained.
