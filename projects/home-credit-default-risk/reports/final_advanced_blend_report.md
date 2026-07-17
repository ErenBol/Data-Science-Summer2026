# Final Advanced Blend Report

Run date: 2026-07-17

## Setup

- CatBoost was trained on the same `advanced_drop_2_weakest_transformed` matrix as the advanced pruned LightGBM.
- LightGBM side uses the untuned advanced-pruned parameters from `final_advanced_pruned_lgbm`, not the later Optuna-tuned model.
- CV uses the same `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` protocol as `run_advanced_relational_cv.py`.
- Target encodings are fold-safe and generated with the same helper functions as the existing advanced CV scripts.
- Blend ratios tested: `50/50, 60/40, 70/30, 80/20` LightGBM/CatBoost.
- Baseline CV ROC-AUC to beat: `0.792549`

## CatBoost CV Metrics

| model | fold | best_iteration | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| catboost_advanced_drop_2 | 1 | 1199 | 363 | 0.792047 | 0.290478 |
| catboost_advanced_drop_2 | 2 | 1197 | 363 | 0.792124 | 0.285188 |
| catboost_advanced_drop_2 | 3 | 1198 | 363 | 0.791568 | 0.281925 |

CatBoost summary:

| mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision |
| --- | --- | --- | --- |
| 0.791913 | 0.000246 | 0.285864 | 0.003524 |

## LightGBM CV Metrics

| model | fold | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- |
| lightgbm_advanced_pruned | 1 | 363 | 0.792999 | 0.293682 |
| lightgbm_advanced_pruned | 2 | 363 | 0.792493 | 0.285731 |
| lightgbm_advanced_pruned | 3 | 363 | 0.792154 | 0.284790 |

LightGBM summary:

| mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision |
| --- | --- | --- | --- |
| 0.792549 | 0.000347 | 0.288067 | 0.003989 |

## Blend CV Metrics

| blend | lightgbm_weight | catboost_weight | fold | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 1 | 0.793890 | 0.294795 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 1 | 0.793925 | 0.295019 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 1 | 0.793850 | 0.295014 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 1 | 0.793671 | 0.294832 |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 2 | 0.793629 | 0.287644 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 2 | 0.793606 | 0.287612 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 2 | 0.793480 | 0.287390 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 2 | 0.793250 | 0.287059 |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 3 | 0.793214 | 0.285951 |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 3 | 0.793212 | 0.286119 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 3 | 0.793105 | 0.286129 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 3 | 0.792894 | 0.285891 |

## Blend CV Summary

| blend | lightgbm_weight | catboost_weight | mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| lgbm_60_catboost_40 | 0.600000 | 0.400000 | 0.793581 | 0.000292 | 0.289583 | 0.003892 |
| lgbm_50_catboost_50 | 0.500000 | 0.500000 | 0.793578 | 0.000278 | 0.289463 | 0.003833 |
| lgbm_70_catboost_30 | 0.700000 | 0.300000 | 0.793478 | 0.000304 | 0.289511 | 0.003925 |
| lgbm_80_catboost_20 | 0.800000 | 0.200000 | 0.793272 | 0.000317 | 0.289261 | 0.003968 |

Best blend: `lgbm_60_catboost_40` with mean CV ROC-AUC `0.793581`.

Selected holdout threshold: `0.663678`

## Holdout Metrics

| model | threshold_strategy | lightgbm_weight | catboost_weight | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_advanced_blend | default_0.5 | 0.600000 | 0.400000 | 0.500000 | 0.795338 | 0.298163 | 0.753264 | 0.201811 | 0.695871 | 0.312882 |
| final_advanced_blend | validation_selected | 0.600000 | 0.400000 | 0.663678 | 0.795338 | 0.298163 | 0.862641 | 0.284548 | 0.463243 | 0.352544 |

## Holdout Comparison

| model | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_advanced_pruned_lgbm | 0.670438 | 0.794353 | 0.296705 | 0.867112 | 0.289391 | 0.443907 | 0.350370 |
| final_advanced_blend | 0.663678 | 0.795338 | 0.298163 | 0.862641 | 0.284548 | 0.463243 | 0.352544 |

Delta, blend minus previous advanced pruned LightGBM:

| metric | delta |
| --- | --- |
| threshold | -0.006760 |
| roc_auc | 0.000985 |
| average_precision | 0.001458 |
| accuracy | -0.004471 |
| precision_class_1 | -0.004843 |
| recall_class_1 | 0.019335 |
| f1_class_1 | 0.002175 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,755 | 5,783 |
| Actual 1 | 2,665 | 2,300 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.90      0.92     56538
           1       0.28      0.46      0.35      4965

    accuracy                           0.86     61503
   macro avg       0.62      0.68      0.64     61503
weighted avg       0.90      0.86      0.88     61503

```

Saved model bundle:

`C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_advanced_blend.joblib`
