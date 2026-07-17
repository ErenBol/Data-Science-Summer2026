# Final Advanced Pruned Optuna Report

Run date: 2026-07-17

## Setup

- Optuna trials: `35`
- Objective: mean 3-fold stratified CV ROC-AUC.
- Feature set: `advanced_drop_2_weakest_transformed`.
- Same fold-safe target encoding and preprocessing setup as `run_advanced_relational_cv.py`.
- Same holdout split and validation-threshold selection flow as `final_advanced_pruned_lgbm`.
- Final selected transformed feature count: `364`
- Selected threshold: `0.656286`

## Best Parameters

| parameter | value |
| --- | --- |
| n_estimators | 1066 |
| learning_rate | 0.040940 |
| num_leaves | 21 |
| max_depth | 9 |
| min_child_samples | 106 |
| subsample | 0.784922 |
| colsample_bytree | 0.747452 |
| reg_alpha | 0.442410 |
| reg_lambda | 13.908076 |
| min_split_gain | 0.282366 |
| class_weight | balanced |
| random_state | 42 |
| n_jobs | -1 |
| verbose | -1 |
| importance_type | gain |

## Best CV Metrics

| mean_cv_roc_auc | std_cv_roc_auc | mean_cv_average_precision | std_cv_average_precision | mean_transformed_feature_count |
| --- | --- | --- | --- | --- |
| 0.792490 | 0.000244 | 0.288340 | 0.002800 | 363.000000 |

## Per-Fold CV Metrics

| fold | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- |
| 1 | 363 | 0.792238 | 0.292163 |
| 2 | 363 | 0.792821 | 0.285537 |
| 3 | 363 | 0.792410 | 0.287319 |

## Holdout Metrics

| model | threshold_strategy | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_advanced_pruned_lgbm_optuna | default_0.5 | 0.500000 | 0.794835 | 0.296204 | 0.764711 | 0.206133 | 0.671501 | 0.315436 |
| final_advanced_pruned_lgbm_optuna | validation_selected | 0.656286 | 0.794835 | 0.296204 | 0.861649 | 0.282683 | 0.464250 | 0.351399 |

## Holdout Comparison

| model | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_advanced_pruned_lgbm | 0.670438 | 0.794353 | 0.296705 | 0.867112 | 0.289391 | 0.443907 | 0.350370 |
| final_advanced_pruned_lgbm_optuna | 0.656286 | 0.794835 | 0.296204 | 0.861649 | 0.282683 | 0.464250 | 0.351399 |

Delta, tuned minus previous advanced pruned:

| metric | delta |
| --- | --- |
| threshold | -0.014151 |
| roc_auc | 0.000482 |
| average_precision | -0.000501 |
| accuracy | -0.005463 |
| precision_class_1 | -0.006707 |
| recall_class_1 | 0.020342 |
| f1_class_1 | 0.001029 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,689 | 5,849 |
| Actual 1 | 2,660 | 2,305 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.90      0.92     56538
           1       0.28      0.46      0.35      4965

    accuracy                           0.86     61503
   macro avg       0.62      0.68      0.64     61503
weighted avg       0.90      0.86      0.88     61503

```

Top Optuna trials:

| trial | mean_roc_auc | mean_average_precision | n_estimators | learning_rate | num_leaves | max_depth | min_child_samples | subsample | colsample_bytree | reg_alpha | reg_lambda | min_split_gain | class_weight | random_state | n_jobs | verbose | importance_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 34 | 0.792490 | 0.288340 | 1066 | 0.040940 | 21 | 9 | 106 | 0.784922 | 0.747452 | 0.442410 | 13.908076 | 0.282366 | balanced | 42 | -1 | -1 | gain |
| 29 | 0.792311 | 0.288393 | 1093 | 0.048674 | 25 | 9 | 83 | 0.810809 | 0.676846 | 1.939091 | 17.309786 | 0.535752 | balanced | 42 | -1 | -1 | gain |
| 30 | 0.792213 | 0.287637 | 967 | 0.050324 | 25 | 10 | 105 | 0.764719 | 0.682304 | 3.242224 | 18.399215 | 0.537256 | balanced | 42 | -1 | -1 | gain |
| 13 | 0.792203 | 0.287469 | 1525 | 0.029188 | 39 | 4 | 146 | 0.847855 | 0.821315 | 0.001300 | 0.418011 | 0.581262 | balanced | 42 | -1 | -1 | gain |
| 2 | 0.792144 | 0.286946 | 1334 | 0.010596 | 39 | 7 | 151 | 0.946294 | 0.719886 | 0.000297 | 0.003236 | 0.046450 | balanced | 42 | -1 | -1 | gain |
| 18 | 0.792127 | 0.287829 | 1410 | 0.010416 | 50 | 7 | 146 | 0.940688 | 0.653093 | 0.009983 | 0.012072 | 0.868227 | balanced | 42 | -1 | -1 | gain |
| 8 | 0.792123 | 0.287221 | 1636 | 0.028088 | 42 | 4 | 122 | 0.831296 | 0.905362 | 0.003516 | 1.786456 | 0.472215 | balanced | 42 | -1 | -1 | gain |
| 17 | 0.792120 | 0.287223 | 942 | 0.026671 | 32 | 10 | 170 | 0.839176 | 0.815936 | 0.000004 | 0.001540 | 0.351967 | balanced | 42 | -1 | -1 | gain |
| 27 | 0.792113 | 0.288502 | 1660 | 0.015838 | 37 | 7 | 214 | 0.997128 | 0.950621 | 0.000006 | 0.000001 | 0.802709 | balanced | 42 | -1 | -1 | gain |
| 24 | 0.792110 | 0.286948 | 1174 | 0.012647 | 34 | 9 | 183 | 0.970368 | 0.856848 | 0.000061 | 0.061688 | 0.990978 | balanced | 42 | -1 | -1 | gain |

Saved model bundle:

`C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_advanced_pruned_lgbm_optuna.joblib`
