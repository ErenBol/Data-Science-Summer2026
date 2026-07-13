from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    scale_numeric: bool = False,
    numeric_add_indicator: bool = True,
    one_hot_sparse_output: bool = True,
    one_hot_min_frequency: int | float | None = None,
    one_hot_max_categories: int | None = None,
) -> ColumnTransformer:
    """Build preprocessing for classification models."""

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=numeric_add_indicator,
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps,
    )

    one_hot_kwargs = {
        "handle_unknown": "ignore",
        "sparse_output": one_hot_sparse_output,
    }

    if one_hot_min_frequency is not None:
        one_hot_kwargs["min_frequency"] = one_hot_min_frequency
        one_hot_kwargs["handle_unknown"] = "infrequent_if_exist"

    if one_hot_max_categories is not None:
        one_hot_kwargs["max_categories"] = one_hot_max_categories
        one_hot_kwargs["handle_unknown"] = "infrequent_if_exist"

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(**one_hot_kwargs),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(numeric_columns),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            ),
        ],
        remainder="drop",
    )
