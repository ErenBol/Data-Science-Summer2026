# Home Credit Default Risk - Profile-Guided Feature Pruning Report

Run date: 2026-07-13

## Purpose

This step exports the pandas profiling report to JSON, reads that JSON, uses it
to inspect raw data quality, then uses LightGBM feature importance and
validation permutation importance to identify low-value features.

Previous notebooks and the existing LightGBM setup were not modified.

## Profiling JSON

Generated artifact:

`reports/application_train_profile.json`

The profiling JSON contains metadata for all 122 raw columns in
`application_train.csv`.

## Buggy-Looking Features From Profiling

The profile scan flagged these main raw-data issues:

- `SK_ID_CURR`: unique identifier, not a modeling feature.
- `TARGET`: label, not a modeling feature.
- Many building/property columns: high missingness, especially
  `COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`, `FONDKAPREMONT_MODE`,
  `LIVINGAPARTMENTS_*`, `FLOORSMIN_*`, and `YEARS_BUILD_*`.
- Near-constant flags: `FLAG_MOBIL`, `FLAG_CONT_MOBILE`, and many
  `FLAG_DOCUMENT_*` columns.
- `DAYS_EMPLOYED`: contains the known sentinel value `365243`.
- `AMT_INCOME_TOTAL`: extreme skew and a very large maximum value.
- `EXT_SOURCE_1`: high missingness, but later model importance shows it is
  still useful.

The current curated feature set already avoids most high-missing building
features and fixes `DAYS_EMPLOYED` in `create_features()`.

## Most Useful Features

LightGBM gain importance and validation permutation importance agreed that the
most useful features are:

| Rank | Feature | Validation Permutation Importance | Gain Share |
|---:|---|---:|---:|
| 1 | `EXT_SOURCE_MEAN` | 0.069517 | 0.407214 |
| 2 | `CREDIT_TERM_APPROX` | 0.018692 | 0.094760 |
| 3 | `AGE_YEARS` | 0.005791 | 0.036427 |
| 4 | `GOODS_CREDIT_RATIO` | 0.004764 | 0.036629 |
| 5 | `AMT_GOODS_PRICE` | 0.004194 | 0.020014 |
| 6 | `NAME_EDUCATION_TYPE` | 0.004085 | 0.020317 |
| 7 | `EXT_SOURCE_3` | 0.004074 | 0.037434 |
| 8 | `CODE_GENDER` | 0.003642 | 0.015639 |
| 9 | `EXT_SOURCE_1` | 0.002899 | 0.018149 |
| 10 | `EXT_SOURCE_MIN` | 0.002276 | 0.054805 |

The most important engineered feature is `EXT_SOURCE_MEAN`, by a large margin.
The next strongest engineered feature is `CREDIT_TERM_APPROX`.

## Low-Relation Features

The conservative low-relation rule selected features with very low gain share
and near-zero or negative validation permutation importance.

Dropped in the first pruning candidate:

- `NAME_HOUSING_TYPE`
- `CNT_FAM_MEMBERS`
- `CNT_CHILDREN`
- `CHILDREN_RATIO`

Additional low-importance features by permutation ranking:

- `AMT_INCOME_TOTAL`
- `INCOME_PER_PERSON`
- `EXT_SOURCE_STD`
- `CREDIT_INCOME_RATIO`

Some of these still have non-trivial gain importance, so they should not be
removed blindly without validation.

## Feature Set Experiments

| Feature Set | Feature Count | Threshold | Validation ROC-AUC | Validation AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `top_24_by_permutation` | 24 | 0.636810 | 0.759686 | 0.240142 | 0.225074 | 0.476838 | 0.305804 |
| `drop_low_relation_features` | 28 | 0.680518 | 0.759578 | 0.239712 | 0.247394 | 0.400302 | 0.305799 |
| `top_20_by_permutation` | 20 | 0.672006 | 0.759492 | 0.239641 | 0.241869 | 0.415660 | 0.305797 |
| `all_32_features` | 32 | 0.679819 | 0.759463 | 0.241413 | 0.248490 | 0.403827 | 0.307663 |

`top_24_by_permutation` produced the best validation ROC-AUC, while the full
32-feature model produced the best validation F1 and average precision.

## Selected Pruned Feature List

The selected pruned list for ranking performance is `top_24_by_permutation`:

```python
[
    "EXT_SOURCE_MEAN",
    "CREDIT_TERM_APPROX",
    "AGE_YEARS",
    "GOODS_CREDIT_RATIO",
    "AMT_GOODS_PRICE",
    "NAME_EDUCATION_TYPE",
    "EXT_SOURCE_3",
    "CODE_GENDER",
    "EXT_SOURCE_1",
    "EXT_SOURCE_MIN",
    "NAME_FAMILY_STATUS",
    "AMT_ANNUITY",
    "EXT_SOURCE_MAX",
    "ANNUITY_INCOME_RATIO",
    "EMPLOYMENT_AGE_RATIO",
    "EXT_SOURCE_COUNT",
    "AMT_CREDIT",
    "OCCUPATION_TYPE",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_2",
    "NAME_INCOME_TYPE",
    "EMPLOYED_YEARS",
    "CREDIT_PER_PERSON",
    "ANNUITY_PER_PERSON",
]
```

## Holdout Test Metrics

Selected model: tuned LightGBM with `top_24_by_permutation`

Selected threshold: `0.636810`

| Threshold Strategy | ROC-AUC | AP | Accuracy | Precision Class 1 | Recall Class 1 | F1 Class 1 |
|---|---:|---:|---:|---:|---:|---:|
| Default 0.5 | 0.764538 | 0.254036 | 0.708014 | 0.171213 | 0.681370 | 0.273661 |
| Validation selected | 0.764538 | 0.254036 | 0.828968 | 0.232569 | 0.486405 | 0.314678 |

Confusion matrix at selected threshold:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 48,569 | 7,969 |
| Actual 1 | 2,550 | 2,415 |

## Recommendation

Use the full 32-feature LightGBM model if the priority is class-1 F1. Use the
24-feature pruned LightGBM model if the priority is ranking quality and a
smaller feature set.

The difference is small:

- Previous 32-feature LightGBM ROC-AUC: `0.764133`
- Pruned 24-feature LightGBM ROC-AUC: `0.764538`
- Previous 32-feature LightGBM F1: `0.315913`
- Pruned 24-feature LightGBM F1: `0.314678`

The pruned model is slightly better for ranking and average precision, but the
full model is slightly better for F1. The most defensible next step is broader
cross-validated LightGBM tuning using both the 32-feature and 24-feature sets.
