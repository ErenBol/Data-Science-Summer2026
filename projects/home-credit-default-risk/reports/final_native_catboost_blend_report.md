# Final Native CatBoost Blend Report

Run date: 2026-07-17

## Setup

- Same holdout split as `final_advanced_blend`: `test_size=0.2`, stratified, `random_state=42`.
- Threshold selected on the same inner validation split from the training portion.
- LightGBM side uses the advanced pruned transformed pipeline with fold-safe target encodings.
- CatBoost side uses native raw categorical handling with `cat_features`; no `TE_*` target-encoded columns are passed to CatBoost.
- Blend weights: `50/50` LightGBM/native CatBoost.
- LightGBM selected transformed features: `364`
- Native CatBoost raw features: `316`
- Native CatBoost categorical features: `14`
- Selected threshold: `0.668978`

## Holdout Metrics

| model | threshold_strategy | lightgbm_weight | catboost_weight | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_native_catboost_blend | default_0.5 | 0.500000 | 0.500000 | 0.500000 | 0.796126 | 0.299074 | 0.751134 | 0.200313 | 0.696073 | 0.311099 |
| final_native_catboost_blend | validation_selected | 0.500000 | 0.500000 | 0.668978 | 0.796126 | 0.299074 | 0.864722 | 0.286006 | 0.451561 | 0.350203 |

## Comparison Against Previous Final Blend

| model | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_advanced_blend | 0.663678 | 0.795338 | 0.298163 | 0.862641 | 0.284548 | 0.463243 | 0.352544 |
| final_native_catboost_blend | 0.668978 | 0.796126 | 0.299074 | 0.864722 | 0.286006 | 0.451561 | 0.350203 |

Delta, native blend minus previous final blend:

| metric | delta |
| --- | --- |
| threshold | 0.005300 |
| roc_auc | 0.000788 |
| average_precision | 0.000911 |
| accuracy | 0.002081 |
| precision_class_1 | 0.001458 |
| recall_class_1 | -0.011682 |
| f1_class_1 | -0.002341 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 50,941 | 5,597 |
| Actual 1 | 2,723 | 2,242 |

Classification report at selected threshold:

```text
              precision    recall  f1-score   support

           0       0.95      0.90      0.92     56538
           1       0.29      0.45      0.35      4965

    accuracy                           0.86     61503
   macro avg       0.62      0.68      0.64     61503
weighted avg       0.90      0.86      0.88     61503

```

Saved model bundle:

`C:\Users\erenb\Desktop\Summer2026\DataScience\projects\home-credit-default-risk\models\final_native_catboost_blend.joblib`
