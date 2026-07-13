# Home Credit Default Risk - Expanded Application Feature Groups Report

Run date: 2026-07-13

## Purpose

This experiment starts from the current curated engineered feature set and tests
additional application-level feature groups suggested by the profile JSON and
feature-importance analysis.

The previous notebooks and model setup were not changed.

## Feature Groups Tested

Added groups:

- contract/context
- time-history
- region/timing
- address mismatch
- bureau request totals
- social-circle ratios
- building/property summaries
- additional likely related features

New engineered features added in `src/features.py`:

- `REGISTRATION_AGE_RATIO`
- `ID_PUBLISH_AGE_RATIO`
- `PHONE_CHANGE_AGE_RATIO`
- `REGION_RATING_DIFF`
- `IS_WEEKEND_APPLICATION`
- `IS_NIGHT_APPLICATION`
- `BUREAU_REQUEST_RECENT_TO_YEAR_RATIO`

## Validation Results

| Feature Set | Raw Features | Engineered Features | Threshold | Validation ROC-AUC | Validation AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| add_through_possible_related_extra | 94 | 128 | 0.688844 | 0.762402 | 0.243919 | 0.257637 | 0.382175 | 0.307786 |
| add_through_social_circle | 46 | 77 | 0.679823 | 0.761799 | 0.243105 | 0.250158 | 0.397533 | 0.307079 |
| add_through_bureau_requests | 42 | 70 | 0.671914 | 0.761662 | 0.243210 | 0.243506 | 0.410624 | 0.305717 |
| add_through_building_summaries | 89 | 123 | 0.678201 | 0.761566 | 0.243647 | 0.249490 | 0.400050 | 0.307320 |
| add_through_address_mismatch | 36 | 61 | 0.677335 | 0.761148 | 0.242680 | 0.247785 | 0.401309 | 0.306391 |
| add_through_region_timing | 30 | 54 | 0.687171 | 0.761005 | 0.241481 | 0.253850 | 0.385952 | 0.306263 |
| baseline_plus_possible_related_extra | 23 | 41 | 0.654929 | 0.760422 | 0.241821 | 0.235364 | 0.446375 | 0.308214 |
| baseline_curated_engineered | 17 | 32 | 0.679819 | 0.759463 | 0.241413 | 0.248490 | 0.403827 | 0.307663 |

The strongest validation ROC-AUC came from adding all requested groups plus the
additional likely related features.

## Selected Feature Set

Selected validation winner:

`add_through_possible_related_extra`

Raw feature count: `94`

Engineered feature count after `create_features()`: `128`

Selected threshold: `0.688844`

This feature set includes the original curated features plus:

- contract/context columns
- registration, ID publish, and phone-change timing
- region and application timing columns
- address mismatch columns
- bureau request columns
- social-circle columns
- building/property numeric columns for summary features
- `OWN_CAR_AGE`
- `TOTALAREA_MODE`
- contact flags

## Holdout Test Metrics

| Threshold Strategy | ROC-AUC | AP | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---:|---:|---:|---:|---:|---:|
| Default 0.5 | 0.769069 | 0.258392 | 0.714876 | 0.174714 | 0.679960 | 0.277997 |
| Validation selected | 0.769069 | 0.258392 | 0.864982 | 0.268735 | 0.390735 | 0.318450 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 51,259 | 5,279 |
| Actual 1 | 3,025 | 1,940 |

## Comparison With Previous Best

Previous tuned LightGBM:

- ROC-AUC: `0.764133`
- Average precision: `0.253209`
- Class-1 F1: `0.315913`

Expanded feature LightGBM:

- ROC-AUC: `0.769069`
- Average precision: `0.258392`
- Class-1 F1: `0.318450`

The expanded feature groups improved ranking quality and thresholded F1.

## Most Important Features In Expanded Model

Top gain-importance signals:

1. `EXT_SOURCE_MEAN`
2. `CREDIT_TERM_APPROX`
3. `EXT_SOURCE_MIN`
4. `EXT_SOURCE_MAX`
5. `GOODS_CREDIT_RATIO`
6. `EXT_SOURCE_3`
7. `AGE_YEARS`
8. `AMT_ANNUITY`
9. `EMPLOYMENT_AGE_RATIO`
10. `AMT_GOODS_PRICE`

Newly added features with visible importance include:

- `OWN_CAR_AGE`
- `DAYS_ID_PUBLISH`
- `ID_PUBLISH_AGE_RATIO`
- `REGION_POPULATION_RELATIVE`
- `REGION_RATING_CLIENT_W_CITY`
- `DAYS_LAST_PHONE_CHANGE`
- `PHONE_CHANGE_AGE_RATIO`
- `BUILDING_FEATURE_MEAN`
- `NAME_CONTRACT_TYPE`
- missing indicator for `OWN_CAR_AGE`

## Recommendation

Use the expanded LightGBM feature set as the current best application-table
model. It improves ROC-AUC, average precision, and F1 over the previous
LightGBM model.

Next, tune LightGBM again on this expanded feature set. The current parameters
were inherited from the smaller 32-feature model, so the expanded feature set
likely deserves its own tuning pass.
