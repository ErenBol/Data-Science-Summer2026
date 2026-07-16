# Advanced Relational Feature CV Report

Run date: 2026-07-16

## Purpose

This experiment tests recent-history relational aggregates, smarter domain aggregates, and fold-safe target/risk encodings with the latest Optuna LightGBM parameters.

Validation uses `StratifiedKFold(n_splits=3)`. Target encodings are fitted inside each fold; training-fold target encodings are generated with an inner 3-fold split to avoid row-level leakage.

## Fold Results

| feature_set | fold | raw_feature_count | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- |
| reduced_top_200 | 1 | 200 | 244 | 0.789806 | 0.288049 |
| reduced_top_200_plus_advanced | 1 | 321 | 365 | 0.792819 | 0.291683 |
| reduced_top_200 | 2 | 200 | 244 | 0.790118 | 0.280199 |
| reduced_top_200_plus_advanced | 2 | 321 | 365 | 0.792308 | 0.285654 |
| reduced_top_200 | 3 | 200 | 244 | 0.788940 | 0.279606 |
| reduced_top_200_plus_advanced | 3 | 321 | 365 | 0.791697 | 0.286258 |

## Summary

| feature_set | average_precision_mean | average_precision_std | raw_feature_count_mean | roc_auc_mean | roc_auc_std | transformed_feature_count_mean |
| --- | --- | --- | --- | --- | --- | --- |
| reduced_top_200 | 0.282618 | 0.003848 | 200.000000 | 0.789622 | 0.000498 | 244.000000 |
| reduced_top_200_plus_advanced | 0.287865 | 0.002711 | 321.000000 | 0.792274 | 0.000459 | 365.000000 |

