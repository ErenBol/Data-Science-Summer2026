# Final Model Summary Report

Run date: 2026-07-17

## Objective

The project predicts Home Credit default risk using application data and relational history tables. The primary ranking metric is ROC-AUC, with average precision and threshold-based class-1 F1 used to check behavior on the imbalanced positive class.

## Data And Feature Sources

- Main table: `application_train.csv` keyed by `SK_ID_CURR`.
- Relational tables: bureau, bureau balance, previous applications, installments, POS cash balance, and credit-card balance.
- Joining strategy: aggregate relational histories to one row per applicant, then left-join to the application row.
- Final native CatBoost side keeps raw categorical application fields plus previous-application categorical summaries; LightGBM side uses the advanced pruned transformed feature pipeline.

## Model Selection Journey

| Stage | Main decision | Result / consequence |
| --- | --- | --- |
| Application-only baseline | Started from engineered application features only | Holdout ROC-AUC `0.769069`; useful but below target. |
| Full relational joins | Added bureau, previous, installments, POS, and credit-card aggregates | Holdout ROC-AUC rose to about `0.790203`; relational history was the largest lift. |
| Feature reduction | Reduced 908 raw / 1884 transformed relational features to top-200 base features | Similar ROC-AUC with far fewer transformed features; reduced over-wide one-hot/aggregate noise. |
| LightGBM tuning | Tuned reduced LightGBM with Optuna | Holdout ROC-AUC `0.791323`; modest but real improvement. |
| Recent/domain features | Added recent-window and domain-specific relational features with fold-safe target encoding | 3-fold CV ROC-AUC improved to `0.792274`; recent behavior helped. |
| Transformed-feature pruning | Dropped weakest transformed occupation one-hot features | Selected `advanced_drop_2_weakest_transformed`; 3-fold CV ROC-AUC `0.792549`. |
| Final advanced pruned LightGBM | Trained holdout model with advanced-drop-2 features | Holdout ROC-AUC `0.794353`, AP `0.296705`, F1 `0.350370`. |
| Advanced LightGBM Optuna | Tuned the advanced-pruned feature set | Holdout ROC-AUC `0.794835`, but AP dipped slightly; not a decisive replacement. |
| CatBoost on transformed features | Tested CatBoost on LightGBM-style transformed matrix | CV ROC-AUC `0.791913`; below LightGBM/blend. |
| Weighted blend | Blended advanced LightGBM with transformed CatBoost | 60/40 blend holdout ROC-AUC `0.795338`, AP `0.298163`, F1 `0.352544`. |
| Second-order recent features | Added compact ratio/trend recent-history features | CV ROC-AUC did not improve (`0.792518` vs `0.792549`), so not promoted. |
| Native categorical CatBoost | Replaced transformed CatBoost side with native categorical CatBoost | CatBoost CV ROC-AUC improved to `0.793058`; 50/50 blend CV ROC-AUC improved to `0.794531`. |
| Final native blend | Trained 50/50 LightGBM/native-CatBoost blend | Holdout ROC-AUC `0.796126`, AP `0.299074`, F1 `0.350203`; best AP and strong ROC-AUC. |
| Stacking | Logistic regression on base OOF predictions plus top raw features | CV ROC-AUC `0.794617` and holdout ROC-AUC `0.796508`, but AP and F1 dropped; not selected as champion. |

## Robustness Check

The final robustness check re-ran 5-fold stratified CV across seeds `42`, `123`, and `2024`, producing 15 fold/seed evaluations per model component.

| model | folds | mean_roc_auc | std_roc_auc | min_roc_auc | max_roc_auc | mean_average_precision | std_average_precision | min_average_precision | max_average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_50_50_blend | 15 | 0.795212 | 0.003537 | 0.790324 | 0.801848 | 0.292133 | 0.006308 | 0.281137 | 0.303106 |
| native_catboost_component | 15 | 0.793633 | 0.003398 | 0.789114 | 0.800027 | 0.288395 | 0.006037 | 0.278541 | 0.298526 |
| lightgbm_component | 15 | 0.793560 | 0.003640 | 0.788319 | 0.800198 | 0.289684 | 0.006194 | 0.278098 | 0.300540 |

Blend advantage over LightGBM component:

| metric | blend_minus_lightgbm |
| --- | --- |
| mean_roc_auc | 0.001653 |
| mean_average_precision | 0.002449 |

Per-fold robustness metrics:

| model | seed | fold | transformed_feature_count | catboost_best_iteration | catboost_categorical_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_component | 42 | 1 | 364.000000 | nan | nan | 0.790282 | 0.287540 |
| native_catboost_component | 42 | 1 | nan | 1193.000000 | 14.000000 | 0.790358 | 0.289012 |
| native_50_50_blend | 42 | 1 | 364.000000 | 1193.000000 | 14.000000 | 0.791893 | 0.291311 |
| lightgbm_component | 42 | 2 | 364.000000 | nan | nan | 0.800198 | 0.300540 |
| native_catboost_component | 42 | 2 | nan | 1195.000000 | 14.000000 | 0.800027 | 0.298526 |
| native_50_50_blend | 42 | 2 | 364.000000 | 1195.000000 | 14.000000 | 0.801848 | 0.303106 |
| lightgbm_component | 42 | 3 | 364.000000 | nan | nan | 0.791243 | 0.281047 |
| native_catboost_component | 42 | 3 | nan | 1199.000000 | 14.000000 | 0.790246 | 0.281439 |
| native_50_50_blend | 42 | 3 | 364.000000 | 1199.000000 | 14.000000 | 0.792253 | 0.283732 |
| lightgbm_component | 42 | 4 | 364.000000 | nan | nan | 0.797043 | 0.298261 |
| native_catboost_component | 42 | 4 | nan | 1196.000000 | 14.000000 | 0.797757 | 0.294826 |
| native_50_50_blend | 42 | 4 | 364.000000 | 1196.000000 | 14.000000 | 0.799035 | 0.299719 |
| lightgbm_component | 42 | 5 | 364.000000 | nan | nan | 0.789324 | 0.282160 |
| native_catboost_component | 42 | 5 | nan | 1199.000000 | 14.000000 | 0.790126 | 0.278541 |
| native_50_50_blend | 42 | 5 | 364.000000 | 1199.000000 | 14.000000 | 0.791293 | 0.283443 |
| lightgbm_component | 123 | 1 | 364.000000 | nan | nan | 0.792166 | 0.289881 |
| native_catboost_component | 123 | 1 | nan | 1199.000000 | 14.000000 | 0.791972 | 0.288168 |
| native_50_50_blend | 123 | 1 | 364.000000 | 1199.000000 | 14.000000 | 0.793712 | 0.292386 |
| lightgbm_component | 123 | 2 | 363.000000 | nan | nan | 0.798065 | 0.289986 |
| native_catboost_component | 123 | 2 | nan | 1180.000000 | 14.000000 | 0.796025 | 0.289972 |
| native_50_50_blend | 123 | 2 | 363.000000 | 1180.000000 | 14.000000 | 0.798679 | 0.292705 |
| lightgbm_component | 123 | 3 | 364.000000 | nan | nan | 0.796206 | 0.292149 |
| native_catboost_component | 123 | 3 | nan | 1194.000000 | 14.000000 | 0.796745 | 0.292379 |
| native_50_50_blend | 123 | 3 | 364.000000 | 1194.000000 | 14.000000 | 0.798062 | 0.295386 |
| lightgbm_component | 123 | 4 | 364.000000 | nan | nan | 0.793871 | 0.289981 |
| native_catboost_component | 123 | 4 | nan | 1199.000000 | 14.000000 | 0.793914 | 0.286547 |
| native_50_50_blend | 123 | 4 | 364.000000 | 1199.000000 | 14.000000 | 0.795521 | 0.291253 |
| lightgbm_component | 123 | 5 | 364.000000 | nan | nan | 0.788319 | 0.285963 |
| native_catboost_component | 123 | 5 | nan | 1199.000000 | 14.000000 | 0.789114 | 0.282031 |
| native_50_50_blend | 123 | 5 | 364.000000 | 1199.000000 | 14.000000 | 0.790324 | 0.287046 |
| lightgbm_component | 2024 | 1 | 364.000000 | nan | nan | 0.794116 | 0.289732 |
| native_catboost_component | 2024 | 1 | nan | 1199.000000 | 14.000000 | 0.794689 | 0.288272 |
| native_50_50_blend | 2024 | 1 | 364.000000 | 1199.000000 | 14.000000 | 0.795998 | 0.291953 |
| lightgbm_component | 2024 | 2 | 364.000000 | nan | nan | 0.793103 | 0.278098 |
| native_catboost_component | 2024 | 2 | nan | 1197.000000 | 14.000000 | 0.793064 | 0.279152 |
| native_50_50_blend | 2024 | 2 | 364.000000 | 1197.000000 | 14.000000 | 0.794732 | 0.281137 |
| lightgbm_component | 2024 | 3 | 364.000000 | nan | nan | 0.798145 | 0.295183 |
| native_catboost_component | 2024 | 3 | nan | 1196.000000 | 14.000000 | 0.797867 | 0.291641 |
| native_50_50_blend | 2024 | 3 | 364.000000 | 1196.000000 | 14.000000 | 0.799699 | 0.296680 |
| lightgbm_component | 2024 | 4 | 364.000000 | nan | nan | 0.790493 | 0.294700 |
| native_catboost_component | 2024 | 4 | nan | 1198.000000 | 14.000000 | 0.791649 | 0.296236 |
| native_50_50_blend | 2024 | 4 | 364.000000 | 1198.000000 | 14.000000 | 0.792669 | 0.299401 |
| lightgbm_component | 2024 | 5 | 364.000000 | nan | nan | 0.790820 | 0.290037 |
| native_catboost_component | 2024 | 5 | nan | 1198.000000 | 14.000000 | 0.790937 | 0.289187 |
| native_50_50_blend | 2024 | 5 | 364.000000 | 1198.000000 | 14.000000 | 0.792470 | 0.292738 |

## Final Holdout Comparison

| model_label | threshold | roc_auc | average_precision | accuracy | precision_class_1 | recall_class_1 | f1_class_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_native_catboost_blend | 0.668978 | 0.796126 | 0.299074 | 0.864722 | 0.286006 | 0.451561 | 0.350203 |
| final_stacked_ensemble | 0.737425 | 0.796508 | 0.296714 | 0.866072 | 0.287145 | 0.444512 | 0.348905 |

## Recommended Final Model

Recommended model: `final_native_catboost_blend`, a 50/50 weighted average of:

- advanced-pruned LightGBM using top-200 reduced base features, recent/domain relational features, fold-safe target encodings, and transformed-feature pruning
- native-categorical CatBoost using the same numeric/relational features plus raw categorical columns passed through `cat_features`

Saved artifact: `models/final_native_catboost_blend.joblib`

Key final holdout metrics:

- ROC-AUC: `0.796126`
- Average precision: `0.299074`
- Accuracy: `0.864722`
- Class-1 precision: `0.286006`
- Class-1 recall: `0.451561`
- Class-1 F1: `0.350203`
- Selected threshold: `0.668978`

## Why This Model Over Stacking

The stacked ensemble had the highest holdout ROC-AUC (`0.796508`), but the gain over the native blend was small (`+0.000382`) and came with worse AP (`0.296714` vs `0.299074`) and worse class-1 F1 (`0.348905` vs `0.350203`). Since the problem is imbalanced, AP and class-1 F1 are important practical checks. The native 50/50 blend is also simpler and more stable operationally than the logistic stack, while preserving nearly the same ROC-AUC and better positive-class quality.

## Final Artifacts

- `models/final_native_catboost_blend.joblib`
- `reports/final_native_catboost_blend_report.md`
- `reports/final_native_blend_robustness_5fold_results.csv`
- `reports/final_native_blend_robustness_5fold_summary.csv`
- `reports/final_model_summary_report.md`
- Supporting experiment scripts in `scripts/` for reproducibility
