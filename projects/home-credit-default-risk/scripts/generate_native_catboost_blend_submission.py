from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_relational_cv import (
    PREVIOUS_TE_COLUMNS,
    build_recent_domain_features,
    build_reduced_matrix,
)
from scripts.run_native_catboost_categorical import (
    build_native_catboost_matrix,
    prepare_native_catboost_frame,
)
from scripts.run_relational_feature_experiments import RAW_DATA_DIR


MODEL_PATH = PROJECT_ROOT / "models" / "final_native_catboost_blend.joblib"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_native_catboost_blend.csv"


def transform_application_te(
    application: pd.DataFrame,
    mappings: dict,
) -> pd.DataFrame:
    features = pd.DataFrame(index=application.index)

    for column, payload in mappings.items():
        mapping = payload["mapping"]
        global_mean = payload["global_mean"]
        features[f"TE_APP_{column}"] = (
            application[column].map(mapping).fillna(global_mean)
        )

    return features


def transform_previous_te(
    application: pd.DataFrame,
    mappings: dict,
) -> pd.DataFrame:
    previous = pd.read_csv(
        RAW_DATA_DIR / "previous_application.csv",
        usecols=["SK_ID_CURR", *PREVIOUS_TE_COLUMNS],
    )
    transform_ids = application[["SK_ID_CURR"]].copy()
    features = pd.DataFrame(index=application.index)

    for column, payload in mappings.items():
        mapping = payload["mapping"]
        global_mean = payload["global_mean"]
        transformed_previous = previous[["SK_ID_CURR", column]].merge(
            transform_ids,
            on="SK_ID_CURR",
            how="inner",
        )
        transformed_previous["ENCODED"] = (
            transformed_previous[column].map(mapping).fillna(global_mean)
        )

        grouped = transformed_previous.groupby("SK_ID_CURR")["ENCODED"]
        by_id = pd.DataFrame(index=transform_ids["SK_ID_CURR"])
        prefix = f"TE_PREV_{column}"
        by_id[f"{prefix}_MEAN"] = grouped.mean()
        by_id[f"{prefix}_MAX"] = grouped.max()
        by_id[f"{prefix}_COUNT"] = grouped.size()
        by_id = by_id.fillna(
            {
                f"{prefix}_MEAN": global_mean,
                f"{prefix}_MAX": global_mean,
                f"{prefix}_COUNT": 0,
            }
        )
        by_id.index = application.index
        features = pd.concat([features, by_id], axis=1)

    return features


def build_lightgbm_test_matrix(
    application_test: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    reduced = build_reduced_matrix(application_test)
    recent_domain = build_recent_domain_features(application_test)
    target_encoding_mappings = bundle["lightgbm_target_encoding_mappings"]
    application_te = transform_application_te(
        application_test,
        target_encoding_mappings["application"],
    )
    previous_te = transform_previous_te(
        application_test,
        target_encoding_mappings["previous"],
    )
    matrix = pd.concat([reduced, recent_domain, application_te, previous_te], axis=1)
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    return matrix.reindex(columns=bundle["lightgbm_raw_feature_columns"])


def build_native_test_matrix(
    application_test: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    matrix = build_native_catboost_matrix(application_test)
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    return matrix.reindex(columns=bundle["native_catboost_feature_columns"])


def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    print(f"Loading model bundle: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)

    print("Reading application_test.csv")
    application_test = pd.read_csv(RAW_DATA_DIR / "application_test.csv")
    ids = application_test["SK_ID_CURR"].copy()

    print("Building LightGBM test matrix")
    lightgbm_matrix = build_lightgbm_test_matrix(application_test, bundle)
    lightgbm_transformed = bundle["lightgbm_preprocessor"].transform(lightgbm_matrix)
    lightgbm_selected = lightgbm_transformed[:, bundle["lightgbm_selected_mask"]]
    lightgbm_probabilities = bundle["lightgbm_model"].predict_proba(
        lightgbm_selected
    )[:, 1]

    print("Building native CatBoost test matrix")
    native_matrix = build_native_test_matrix(application_test, bundle)
    native_frame, _, _ = prepare_native_catboost_frame(native_matrix)
    catboost_probabilities = bundle["native_catboost_model"].predict_proba(
        native_frame
    )[:, 1]

    probabilities = (
        bundle["lightgbm_weight"] * lightgbm_probabilities
        + bundle["catboost_weight"] * catboost_probabilities
    )
    submission = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": probabilities,
        }
    )

    if len(submission) != len(application_test):
        raise ValueError(
            f"Submission row count {len(submission)} does not match "
            f"application_test row count {len(application_test)}."
        )
    if not submission["TARGET"].between(0, 1).all():
        bad_count = int((~submission["TARGET"].between(0, 1)).sum())
        raise ValueError(f"Found {bad_count} TARGET values outside [0, 1].")
    if list(submission.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Submission columns are not exactly SK_ID_CURR and TARGET.")
    if not submission["SK_ID_CURR"].equals(ids):
        raise ValueError("Submission ID order does not match application_test.csv.")

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH}")
    print(f"Rows: {len(submission)}")
    print(
        "TARGET range: "
        f"{submission['TARGET'].min():.8f} - {submission['TARGET'].max():.8f}"
    )


if __name__ == "__main__":
    main()
