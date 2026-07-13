from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_relational_feature_experiments import (
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TEST_SIZE,
    VALIDATION_SIZE,
    align_to_application,
    build_application_features,
    build_feature_matrix,
    get_feature_importance,
    get_positive_probabilities,
)
from scripts.run_relational_feature_pruning import (
    RELATIONAL_GROUPS,
    Variant,
    as_markdown,
    build_lgbm_pipeline,
    domain_compact_columns,
    evaluate_variant,
    top_gain_columns,
)
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"


def summarize_profile_jsons() -> pd.DataFrame:
    rows = []

    for path in sorted(REPORTS_DIR.glob("*_profile.json")):
        table = path.name.removesuffix("_profile.json")
        if table == "application_train":
            continue

        with path.open("r", encoding="utf-8") as file:
            profile = json.load(file)

        variables = profile.get("variables", {})
        for name, metadata in variables.items():
            rows.append(
                {
                    "table": table,
                    "feature": name,
                    "type": metadata.get("type", ""),
                    "p_missing": metadata.get("p_missing", 0) or 0,
                    "p_distinct": metadata.get("p_distinct", 0) or 0,
                    "n_distinct": metadata.get("n_distinct", 0) or 0,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORTS_DIR / "relational_profile_feature_review.csv", index=False)

    return summary


def feature_count_diagnostics(source_gain: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for prefix, label in [
        ("BURO", "bureau + bureau_balance"),
        ("PREV", "previous_application"),
        ("INST", "installments_payments"),
        ("POS", "POS_CASH_balance"),
        ("CC", "credit_card_balance"),
    ]:
        group = source_gain[source_gain["source_column"].str.startswith(prefix)]
        rows.append(
            {
                "feature_group": label,
                "raw_source_columns": len(group),
                "positive_gain_columns": int(
                    (group["positive_transformed_feature_count"] > 0).sum()
                ),
                "zero_gain_columns": int(
                    (group["positive_transformed_feature_count"] == 0).sum()
                ),
                "total_gain": group["total_gain"].sum(),
            }
        )

    app = source_gain[
        ~source_gain["source_column"].str.startswith(
            ("BURO", "PREV", "INST", "POS", "CC")
        )
    ]
    rows.insert(
        0,
        {
            "feature_group": "application engineered",
            "raw_source_columns": len(app),
            "positive_gain_columns": int(
                (app["positive_transformed_feature_count"] > 0).sum()
            ),
            "zero_gain_columns": int(
                (app["positive_transformed_feature_count"] == 0).sum()
            ),
            "total_gain": app["total_gain"].sum(),
        },
    )

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(
        REPORTS_DIR / "relational_feature_count_diagnostics.csv",
        index=False,
    )

    return diagnostics


def build_joined_matrix() -> tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index, pd.Index]:
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    train_index, test_index = train_test_split(
        train_df.index,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_full_index, valid_index = train_test_split(
        train_index,
        test_size=VALIDATION_SIZE,
        stratify=y.loc[train_index],
        random_state=RANDOM_STATE,
    )

    app_features = build_application_features(train_df)
    aligned_groups = {}
    for group_name in RELATIONAL_GROUPS:
        features = pd.read_pickle(PROCESSED_DIR / f"{group_name}_features.pkl")
        aligned_groups[group_name] = align_to_application(train_df, features)
        del features
        gc.collect()

    X_full = build_feature_matrix(app_features, aligned_groups, RELATIONAL_GROUPS)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)

    return X_full, y, train_index, train_full_index, valid_index


def write_report(
    results: pd.DataFrame,
    final_metrics: pd.DataFrame,
    final_confusion_matrix: np.ndarray,
    final_classification: str,
    final_importance: pd.DataFrame,
    profile_summary: pd.DataFrame,
    count_diagnostics: pd.DataFrame,
    saved_model_path: Path | None,
    save_error: str | None,
) -> None:
    profile_view = (
        profile_summary.sort_values(["p_missing", "p_distinct"], ascending=False)
        .head(40)
        .copy()
    )
    count_view = count_diagnostics.sort_values("raw_source_columns", ascending=False)

    lines = [
        "# Home Credit Relational Feature Reduction Report",
        "",
        "Run date: 2026-07-13",
        "",
        "## Why The Feature Count Grew",
        "",
        "The feature count grew from about 100 application features to 908 joined raw "
        "features because each relational table has many rows per applicant. The "
        "pipeline first aggregates those histories to `SK_ID_CURR`; each numeric "
        "field creates several statistics such as mean, max, min, and sum, and "
        "each categorical field creates count/rate columns for its observed values. "
        "`previous_application` is the largest source because it has many product, "
        "status, purpose, channel, and category fields.",
        "",
        as_markdown(count_view),
        "",
        "## Profile JSON Review",
        "",
        "The profile JSONs confirm which fields are structurally risky: high-missing "
        "fields are useful only when aggregated carefully, and high-cardinality IDs "
        "must never be used directly. The modeling pipeline uses `SK_ID_CURR` only "
        "for joins and does not train on raw ID columns.",
        "",
        as_markdown(profile_view),
        "",
        "## Reduction Validation Results",
        "",
        as_markdown(results),
        "",
        "## Final Holdout Metrics",
        "",
        as_markdown(final_metrics),
        "",
        "Confusion matrix at selected threshold:",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---:|---:|",
        f"| Actual 0 | {final_confusion_matrix[0, 0]:,} | {final_confusion_matrix[0, 1]:,} |",
        f"| Actual 1 | {final_confusion_matrix[1, 0]:,} | {final_confusion_matrix[1, 1]:,} |",
        "",
        "Classification report at selected threshold:",
        "",
        "```text",
        final_classification,
        "```",
        "",
        "## Final Feature Importance",
        "",
        as_markdown(final_importance.head(50)),
        "",
        "## Model Saving",
        "",
    ]

    if saved_model_path is not None:
        lines.append(f"Saved model: `{saved_model_path}`")
    else:
        lines.append(f"Model save failed: `{save_error}`")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use the smallest model whose validation and holdout metrics remain close "
            "to the full relational model. Further feature count reduction should be "
            "done by changing the relational aggregation recipe itself, especially "
            "by removing rare previous-application categorical count/rate features, "
            "not by adding more one-hot controls.",
        ]
    )

    (REPORTS_DIR / "relational_feature_reduction_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("Reviewing profile JSON files")
    profile_summary = summarize_profile_jsons()
    source_gain = pd.read_csv(REPORTS_DIR / "relational_source_column_gain.csv")
    count_diagnostics = feature_count_diagnostics(source_gain)

    print("Building joined feature matrix")
    X_full, y, train_index, train_full_index, valid_index = build_joined_matrix()
    test_index = y.index.difference(train_index)

    y_train = y.loc[train_full_index]
    y_valid = y.loc[valid_index]
    y_train_full = y.loc[train_index]
    y_test = y.loc[test_index]

    variants = [
        Variant(
            name="domain_250_no_missing_indicators",
            columns=domain_compact_columns(source_gain, 250),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        ),
        Variant(
            name="domain_200_no_missing_indicators",
            columns=domain_compact_columns(source_gain, 200),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        ),
        Variant(
            name="domain_150_no_missing_indicators",
            columns=domain_compact_columns(source_gain, 150),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        ),
        Variant(
            name="top_250_no_missing_indicators",
            columns=top_gain_columns(source_gain, 250),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        ),
        Variant(
            name="top_200_no_missing_indicators",
            columns=top_gain_columns(source_gain, 200),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        ),
        Variant(
            name="top_150_no_missing_indicators",
            columns=top_gain_columns(source_gain, 150),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        ),
    ]

    rows = []
    importance_frames = []
    for variant in variants:
        print(f"Training {variant.name}: {len(variant.columns)} raw columns")
        row, importance = evaluate_variant(
            variant=variant,
            X_full=X_full,
            y_train=y_train,
            y_valid=y_valid,
            train_full_index=train_full_index,
            valid_index=valid_index,
        )
        rows.append(row)
        top_importance = importance.head(50).copy()
        top_importance.insert(0, "feature_set", variant.name)
        importance_frames.append(top_importance)

        pd.DataFrame(rows).sort_values(
            ["validation_roc_auc", "validation_f1_class_1"],
            ascending=False,
        ).to_csv(REPORTS_DIR / "relational_feature_reduction_results.csv", index=False)
        pd.concat(importance_frames, ignore_index=True).to_csv(
            REPORTS_DIR / "relational_reduction_importance_by_experiment.csv",
            index=False,
        )

        del importance, top_importance
        gc.collect()

    results = pd.DataFrame(rows).sort_values(
        ["validation_roc_auc", "validation_f1_class_1"],
        ascending=False,
    )
    results.to_csv(REPORTS_DIR / "relational_feature_reduction_results.csv", index=False)

    best_name = str(results.iloc[0]["feature_set"])
    best_variant = next(variant for variant in variants if variant.name == best_name)
    best_threshold = float(results.iloc[0]["threshold"])
    print(f"Final training selected reduction: {best_name}")

    X_best = X_full[best_variant.columns]
    X_train_full = X_best.loc[train_index]
    X_test = X_best.loc[test_index]

    final_pipeline = build_lgbm_pipeline(
        X_train=X_train_full,
        numeric_add_indicator=best_variant.numeric_add_indicator,
        one_hot_min_frequency=best_variant.one_hot_min_frequency,
        one_hot_max_categories=best_variant.one_hot_max_categories,
    )
    final_pipeline.fit(X_train_full, y_train_full)
    probabilities = get_positive_probabilities(final_pipeline, X_test)

    final_metrics = pd.DataFrame(
        [
            {
                "feature_set": best_name,
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y_test, probabilities, 0.5),
            },
            {
                "feature_set": best_name,
                "threshold_strategy": "validation_selected",
                **evaluate_probabilities(y_test, probabilities, best_threshold),
            },
        ]
    )
    final_metrics.to_csv(
        REPORTS_DIR / "relational_reduced_final_test_metrics.csv",
        index=False,
    )

    predictions = (probabilities >= best_threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(REPORTS_DIR / "relational_reduced_final_confusion_matrix.csv")

    final_classification = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
    (REPORTS_DIR / "relational_reduced_final_classification_report.txt").write_text(
        final_classification,
        encoding="utf-8",
    )

    final_importance = get_feature_importance(final_pipeline)
    final_importance.to_csv(
        REPORTS_DIR / "relational_reduced_final_feature_importance.csv",
        index=False,
    )

    model_path = MODELS_DIR / "final_relational_reduced_lgbm.joblib"
    saved_model_path = None
    save_error = None
    try:
        joblib.dump(final_pipeline, model_path, compress=3)
        saved_model_path = model_path
    except OSError as error:
        save_error = str(error)

    write_report(
        results=results,
        final_metrics=final_metrics,
        final_confusion_matrix=matrix,
        final_classification=final_classification,
        final_importance=final_importance,
        profile_summary=profile_summary,
        count_diagnostics=count_diagnostics,
        saved_model_path=saved_model_path,
        save_error=save_error,
    )

    print(results.to_string(index=False))
    print(final_metrics.to_string(index=False))
    if save_error:
        print(f"Model save failed: {save_error}")
    else:
        print(f"Saved model to {saved_model_path}")


if __name__ == "__main__":
    main()
