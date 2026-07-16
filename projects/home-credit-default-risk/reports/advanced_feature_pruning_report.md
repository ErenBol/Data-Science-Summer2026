# Advanced Feature Pruning Report

Run date: 2026-07-16

## Purpose

This experiment prunes transformed features created by the preprocessing pipeline, not only raw input columns. One-hot encoded categories and numeric transformed features are evaluated by LightGBM gain importance.

The pruning search uses the first stratified fold as a development fold. Each iteration drops the lowest-gain transformed feature batch and retrains. The maximum number of pruning iterations is `20`.

## Iterative Search

| iteration | candidate_action | dropped_count | remaining_features | roc_auc | average_precision | accepted | candidate_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | 0 | 365 | 0.792819 | 0.291683 | True | advanced_all_transformed |
| 1 | drop_cumulative_weakest_transformed_features | 1 | 364 | 0.792981 | 0.292900 | True | drop_1_weakest_transformed |
| 2 | drop_cumulative_weakest_transformed_features | 2 | 363 | 0.792999 | 0.293682 | True | drop_2_weakest_transformed |
| 3 | drop_cumulative_weakest_transformed_features | 5 | 360 | 0.792594 | 0.291832 | True | drop_5_weakest_transformed |
| 4 | drop_cumulative_weakest_transformed_features | 10 | 355 | 0.792746 | 0.292599 | True | drop_10_weakest_transformed |
| 5 | drop_cumulative_weakest_transformed_features | 15 | 350 | 0.792563 | 0.292146 | True | drop_15_weakest_transformed |
| 6 | drop_cumulative_weakest_transformed_features | 20 | 345 | 0.792533 | 0.293049 | True | drop_20_weakest_transformed |
| 7 | drop_cumulative_weakest_transformed_features | 25 | 340 | 0.792330 | 0.293289 | True | drop_25_weakest_transformed |
| 8 | drop_cumulative_weakest_transformed_features | 35 | 330 | 0.793022 | 0.293227 | True | drop_35_weakest_transformed |
| 9 | drop_cumulative_weakest_transformed_features | 50 | 315 | 0.792718 | 0.291301 | True | drop_50_weakest_transformed |
| 10 | drop_cumulative_weakest_transformed_features | 75 | 290 | 0.792466 | 0.292327 | True | drop_75_weakest_transformed |
| 11 | drop_cumulative_weakest_transformed_features | 100 | 265 | 0.792740 | 0.292785 | True | drop_100_weakest_transformed |
| 12 | drop_cumulative_weakest_transformed_features | 125 | 240 | 0.792533 | 0.292118 | True | drop_125_weakest_transformed |
| 13 | drop_cumulative_weakest_transformed_features | 150 | 215 | 0.792439 | 0.291416 | True | drop_150_weakest_transformed |

## Dropped Features In Selected Set

| iteration | transformed_feature | accepted |
| --- | --- | --- |
| 2 | categorical__OCCUPATION_TYPE_infrequent_sklearn | True |
| 2 | categorical__OCCUPATION_TYPE_Security staff | True |

Selected CV feature set: `advanced_drop_2_weakest_transformed`

Selected transformed feature count after pruning: `363`

## Final 3-Fold CV Comparison

| feature_set | fold | transformed_feature_count | roc_auc | average_precision |
| --- | --- | --- | --- | --- |
| advanced_all_transformed | 1 | 365 | 0.792819 | 0.291683 |
| advanced_all_transformed | 2 | 365 | 0.792308 | 0.285654 |
| advanced_all_transformed | 3 | 365 | 0.791697 | 0.286258 |
| advanced_drop_35_weakest_transformed | 1 | 330 | 0.793022 | 0.293227 |
| advanced_drop_35_weakest_transformed | 2 | 330 | 0.792757 | 0.286166 |
| advanced_drop_35_weakest_transformed | 3 | 330 | 0.791589 | 0.285353 |
| advanced_drop_150_weakest_transformed | 1 | 215 | 0.792439 | 0.291416 |
| advanced_drop_150_weakest_transformed | 2 | 215 | 0.792349 | 0.285090 |
| advanced_drop_150_weakest_transformed | 3 | 215 | 0.791795 | 0.286216 |
| advanced_drop_2_weakest_transformed | 1 | 363 | 0.792999 | 0.293682 |
| advanced_drop_2_weakest_transformed | 2 | 363 | 0.792493 | 0.285731 |
| advanced_drop_2_weakest_transformed | 3 | 363 | 0.792154 | 0.284790 |

## CV Summary

| feature_set | average_precision_mean | average_precision_std | roc_auc_mean | roc_auc_std | transformed_feature_count_mean |
| --- | --- | --- | --- | --- | --- |
| advanced_all_transformed | 0.287865 | 0.002711 | 0.792274 | 0.000459 | 365.000000 |
| advanced_drop_150_weakest_transformed | 0.287574 | 0.002755 | 0.792194 | 0.000284 | 215.000000 |
| advanced_drop_2_weakest_transformed | 0.288067 | 0.003989 | 0.792549 | 0.000347 | 363.000000 |
| advanced_drop_35_weakest_transformed | 0.288248 | 0.003536 | 0.792456 | 0.000622 | 330.000000 |

