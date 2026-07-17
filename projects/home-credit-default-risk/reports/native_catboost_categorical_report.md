# Native CatBoost Categorical Report

Run date: 2026-07-17

## Setup

- Trained CatBoost on the raw advanced feature frame instead of the LightGBM-style transformed matrix.
- Raw application categorical columns and raw previous-application categorical replacements are passed through `cat_features`.
- Fold-safe `TE_*` target-encoding columns are intentionally omitted from the native CatBoost frame.
- Numeric top-200 reduced features and advanced recent/domain relational features are kept unchanged.
- CV protocol: same 3-fold stratified split as `run_advanced_relational_cv.py`.

## Native CatBoost CV Results

| model | fold | raw_feature_count | categorical_feature_count | best_iteration | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| native_catboost_categorical | 1 | 316 | 14 | 1199 | 0.792618 | 0.291553 |
| native_catboost_categorical | 2 | 316 | 14 | 1199 | 0.793790 | 0.286989 |
| native_catboost_categorical | 3 | 316 | 14 | 1199 | 0.792765 | 0.283059 |

## CatBoost Comparison

| model | mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision |
| --- | --- | --- | --- | --- |
| previous_transformed_catboost | 0.791913 | nan | nan | nan |
| native_catboost_categorical | 0.793058 | 0.000521 | 0.287201 | 0.003471 |

Delta versus previous transformed CatBoost:

| metric | delta |
| --- | --- |
| mean_cv_roc_auc | 0.001145 |

Native CatBoost improved over the previous CatBoost result, so the LightGBM+CatBoost blend search was rerun.

## LightGBM CV Side

| model | fold | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- |
| lightgbm_advanced_pruned | 1 | 363 | 0.792999 | 0.293682 |
| lightgbm_advanced_pruned | 2 | 363 | 0.792493 | 0.285731 |
| lightgbm_advanced_pruned | 3 | 363 | 0.792154 | 0.284790 |

## Blend CV Results

| blend | lightgbm_weight | catboost_weight | fold | catboost_best_iteration | catboost_categorical_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 1 | 1199 | 14 | 0.794578 | 0.295940 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 1 | 1199 | 14 | 0.794540 | 0.296009 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 1 | 1199 | 14 | 0.794360 | 0.295904 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 1 | 1199 | 14 | 0.794041 | 0.295492 |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 2 | 1199 | 14 | 0.794808 | 0.289420 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 2 | 1199 | 14 | 0.794603 | 0.289196 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 2 | 1199 | 14 | 0.794267 | 0.288740 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 2 | 1199 | 14 | 0.793805 | 0.288010 |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 3 | 1199 | 14 | 0.794207 | 0.287232 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 3 | 1199 | 14 | 0.794067 | 0.287249 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 3 | 1199 | 14 | 0.793788 | 0.287008 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 3 | 1199 | 14 | 0.793376 | 0.286587 |

## Blend CV Summary

| blend | lightgbm_weight | catboost_weight | mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 0.794531 | 0.000248 | 0.290864 | 0.003699 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 0.794403 | 0.000239 | 0.290818 | 0.003756 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 0.794138 | 0.000251 | 0.290550 | 0.003851 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 0.793741 | 0.000275 | 0.290030 | 0.003906 |

Best native-CatBoost blend: `lgbm_50_catboost_50` with mean CV ROC-AUC `0.794531`.
Current final advanced blend CV ROC-AUC baseline: `0.793581`.
Delta: `0.000950`.
