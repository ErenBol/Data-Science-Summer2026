from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_relational_feature_experiments import (
    LIGHTGBM_PARAMS,
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
from src.preprocessing import build_preprocessor
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
RELATIONAL_GROUPS = [
    "bureau",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]

DOMAIN_KEEP_PATTERNS = [
    "EXT_SOURCE",
    "CREDIT_TERM",
    "GOODS_CREDIT",
    "ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS",
    "DAYS_EMPLOYED",
    "EMPLOYMENT",
    "AGE_YEARS",
    "OWN_CAR_AGE",
    "REGION_POPULATION",
    "BURO_RECORD_COUNT",
    "BURO_DEBT_CREDIT_RATIO",
    "BURO_OVERDUE",
    "BURO_DAYS_CREDIT",
    "BURO_AMT_CREDIT_SUM",
    "BURO_ACTIVE",
    "BB_BAD_STATUS",
    "BB_STATUS",
    "PREV_RECORD_COUNT",
    "PREV_CREDIT_APPLICATION_RATIO",
    "PREV_ANNUITY_CREDIT_RATIO",
    "PREV_DAYS_DECISION",
    "PREV_DAYS_LAST_DUE",
    "PREV_NAME_CONTRACT_STATUS",
    "PREV_AMT_APPLICATION",
    "PREV_AMT_CREDIT",
    "PREV_CNT_PAYMENT",
    "INST_RECORD_COUNT",
    "INST_LATE_PAYMENT",
    "INST_UNDERPAYMENT",
    "INST_PAYMENT_DELAY",
    "INST_PAYMENT_RATIO",
    "INST_PAYMENT_DIFF",
    "INST_AMT_PAYMENT",
    "INST_AMT_INSTALMENT",
    "POS_RECORD_COUNT",
    "POS_COMPLETION",
    "POS_SK_DPD",
    "POS_CNT_INSTALMENT",
    "POS_STATUS",
    "CC_RECORD_COUNT",
    "CC_UTILIZATION",
    "CC_PAYMENT_MIN",
    "CC_DRAWING_LIMIT",
    "CC_AMT_BALANCE",
    "CC_CNT_DRAWINGS",
    "CC_SK_DPD",
]


@dataclass(frozen=True)
class Variant:
    name: str
    columns: list[str]
    numeric_add_indicator: bool = True
    one_hot_min_frequency: int | float | None = None
    one_hot_max_categories: int | None = None


def as_markdown(data: pd.DataFrame) -> str:
    columns = data.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_lgbm_pipeline(
    X_train: pd.DataFrame,
    numeric_add_indicator: bool,
    one_hot_min_frequency: int | float | None,
    one_hot_max_categories: int | None,
) -> Pipeline:
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_add_indicator=numeric_add_indicator,
        one_hot_sparse_output=True,
        one_hot_min_frequency=one_hot_min_frequency,
        one_hot_max_categories=one_hot_max_categories,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**LIGHTGBM_PARAMS)),
        ]
    )


def source_column_from_transformed(
    transformed_feature: str,
    categorical_columns: list[str],
) -> str:
    if transformed_feature.startswith("numeric__missingindicator_"):
        return transformed_feature.removeprefix("numeric__missingindicator_")

    if transformed_feature.startswith("numeric__"):
        return transformed_feature.removeprefix("numeric__")

    if transformed_feature.startswith("categorical__"):
        encoded = transformed_feature.removeprefix("categorical__")
        for column in sorted(categorical_columns, key=len, reverse=True):
            if encoded == column or encoded.startswith(f"{column}_"):
                return column

    return transformed_feature


def build_source_gain(
    X_full: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    categorical_columns = X_full.select_dtypes(exclude="number").columns.tolist()
    gain = importance.copy()
    gain["source_column"] = gain["transformed_feature"].map(
        lambda feature: source_column_from_transformed(feature, categorical_columns)
    )

    source_gain = (
        gain.groupby("source_column", as_index=False)
        .agg(
            total_gain=("gain_importance", "sum"),
            max_gain=("gain_importance", "max"),
            transformed_feature_count=("transformed_feature", "count"),
            positive_transformed_feature_count=(
                "gain_importance",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values(["total_gain", "max_gain"], ascending=False)
    )

    source_gain["source_column_exists"] = source_gain["source_column"].isin(
        X_full.columns
    )

    return source_gain


def top_gain_columns(source_gain: pd.DataFrame, limit: int) -> list[str]:
    return (
        source_gain[source_gain["source_column_exists"]]
        .head(limit)["source_column"]
        .tolist()
    )


def positive_gain_columns(source_gain: pd.DataFrame) -> list[str]:
    rows = source_gain[
        source_gain["source_column_exists"]
        & (source_gain["positive_transformed_feature_count"] > 0)
    ]

    return rows["source_column"].tolist()


def domain_compact_columns(source_gain: pd.DataFrame, limit: int) -> list[str]:
    candidates = source_gain[
        source_gain["source_column_exists"]
        & (source_gain["positive_transformed_feature_count"] > 0)
    ].copy()
    candidates["domain_keep"] = candidates["source_column"].map(
        lambda column: any(pattern in column for pattern in DOMAIN_KEEP_PATTERNS)
    )

    selected = candidates[candidates["domain_keep"]]
    if len(selected) < limit:
        selected = pd.concat(
            [
                selected,
                candidates[~candidates["domain_keep"]].head(limit - len(selected)),
            ]
        )

    return selected.head(limit)["source_column"].tolist()


def evaluate_variant(
    variant: Variant,
    X_full: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    train_full_index: pd.Index,
    valid_index: pd.Index,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    X_variant = X_full[variant.columns]
    X_train = X_variant.loc[train_full_index]
    X_valid = X_variant.loc[valid_index]

    pipeline = build_lgbm_pipeline(
        X_train=X_train,
        numeric_add_indicator=variant.numeric_add_indicator,
        one_hot_min_frequency=variant.one_hot_min_frequency,
        one_hot_max_categories=variant.one_hot_max_categories,
    )
    pipeline.fit(X_train, y_train)

    valid_probabilities = get_positive_probabilities(pipeline, X_valid)
    threshold_info = find_best_threshold(y_valid, valid_probabilities)
    metrics = evaluate_probabilities(
        y_true=y_valid,
        probabilities=valid_probabilities,
        threshold=threshold_info["threshold"],
    )
    transformed_feature_count = len(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    importance = get_feature_importance(pipeline)

    result = {
        "feature_set": variant.name,
        "raw_feature_count": len(variant.columns),
        "transformed_feature_count": transformed_feature_count,
        "numeric_add_indicator": variant.numeric_add_indicator,
        "one_hot_min_frequency": variant.one_hot_min_frequency,
        "one_hot_max_categories": variant.one_hot_max_categories,
        "threshold": threshold_info["threshold"],
        "validation_roc_auc": metrics["roc_auc"],
        "validation_average_precision": metrics["average_precision"],
        "validation_accuracy": metrics["accuracy"],
        "validation_precision_class_1": metrics["precision_class_1"],
        "validation_recall_class_1": metrics["recall_class_1"],
        "validation_f1_class_1": metrics["f1_class_1"],
    }

    return result, importance


def write_report(
    results: pd.DataFrame,
    final_metrics: pd.DataFrame,
    final_confusion_matrix: np.ndarray,
    final_classification_report: str,
    top_importance: pd.DataFrame,
    source_gain: pd.DataFrame,
) -> None:
    compact_metrics_path = (
        REPORTS_DIR / "relational_domain_compact_final_test_metrics.csv"
    )
    compact_matrix_path = (
        REPORTS_DIR / "relational_domain_compact_final_confusion_matrix.csv"
    )
    compact_importance_path = (
        REPORTS_DIR / "relational_domain_compact_final_feature_importance.csv"
    )
    compact_lines = []
    if compact_metrics_path.exists():
        compact_metrics = pd.read_csv(compact_metrics_path)
        compact_lines.extend(
            [
                "## Compact Domain Holdout Check",
                "",
                "The compact domain variant is shown separately because it gives the best feature-count tradeoff.",
                "",
                as_markdown(compact_metrics),
                "",
            ]
        )

        if compact_matrix_path.exists():
            compact_matrix = pd.read_csv(compact_matrix_path, index_col=0).to_numpy()
            compact_lines.extend(
                [
                    "Compact domain confusion matrix at selected threshold:",
                    "",
                    "| | Predicted 0 | Predicted 1 |",
                    "|---|---:|---:|",
                    f"| Actual 0 | {compact_matrix[0, 0]:,} | {compact_matrix[0, 1]:,} |",
                    f"| Actual 1 | {compact_matrix[1, 0]:,} | {compact_matrix[1, 1]:,} |",
                    "",
                ]
            )

        if compact_importance_path.exists():
            compact_importance = pd.read_csv(compact_importance_path).head(40)
            compact_lines.extend(
                [
                    "Compact domain top feature importances:",
                    "",
                    as_markdown(compact_importance),
                    "",
                ]
            )

    lines = [
        "# Home Credit Relational Feature Pruning Report",
        "",
        "Run date: 2026-07-13",
        "",
        "## Purpose",
        "",
        "This experiment keeps the same train/validation/test split and the same "
        "LightGBM hyperparameters, then tests whether the joined relational model "
        "can be made smaller by dropping zero/low-gain raw columns and reducing "
        "pipeline expansion.",
        "",
        "Controls tested:",
        "",
        "- grouped application one-hot encoding with `min_frequency=1000` and `max_categories=20`",
        "- removing numeric imputer missing-indicator expansion",
        "- keeping only raw columns mapped to positive transformed gain",
        "- keeping top raw columns by summed transformed gain",
        "- keeping a domain-compact subset from repayment, bureau, previous-loan, POS, credit-card, and core application groups",
        "",
        "## Validation Results",
        "",
        as_markdown(results),
        "",
        "## Final Holdout Metrics",
        "",
        as_markdown(final_metrics),
        "",
        *compact_lines,
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
        final_classification_report,
        "```",
        "",
        "## Top Final Feature Importances",
        "",
        as_markdown(top_importance),
        "",
        "## Raw Column Gain Summary",
        "",
        as_markdown(source_gain.head(40)),
        "",
        "## Findings",
        "",
        "- `create_features()` is appropriate for the application table only. It should not be expanded to relational tables; the relational tables need grouped aggregations before joining.",
        "- The >1000 transformed-feature count mostly came from relational category count/rate aggregates plus numeric missing indicators, not just application one-hot encoding.",
        "- One-hot grouping helps control rare application categories, but it is not enough by itself because application categoricals are a small share of the transformed matrix.",
        "- Gain-based raw-column pruning is the direct way to reduce the joined feature matrix while preserving the same modeling setup.",
    ]

    (REPORTS_DIR / "relational_feature_pruning_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("Reading application_train.csv")
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

    y_train = y.loc[train_full_index]
    y_valid = y.loc[valid_index]
    y_train_full = y.loc[train_index]
    y_test = y.loc[test_index]

    print("Building joined relational matrix from cached aggregates")
    app_features = build_application_features(train_df)
    aligned_groups = {}
    for group_name in RELATIONAL_GROUPS:
        features = pd.read_pickle(PROCESSED_DIR / f"{group_name}_features.pkl")
        aligned_groups[group_name] = align_to_application(train_df, features)
        del features
        gc.collect()

    X_full = build_feature_matrix(app_features, aligned_groups, RELATIONAL_GROUPS)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)

    previous_importance = pd.read_csv(
        REPORTS_DIR / "relational_final_feature_importance.csv"
    )
    source_gain = build_source_gain(X_full, previous_importance)
    source_gain.to_csv(
        REPORTS_DIR / "relational_source_column_gain.csv",
        index=False,
    )

    all_columns = X_full.columns.tolist()
    positive_columns = positive_gain_columns(source_gain)

    variants = [
        Variant(
            name="all_features_no_missing_indicators",
            columns=all_columns,
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=20,
        ),
        Variant(
            name="positive_gain_raw_grouped_ohe",
            columns=positive_columns,
            numeric_add_indicator=True,
            one_hot_min_frequency=1000,
            one_hot_max_categories=20,
        ),
        Variant(
            name="positive_gain_raw_no_missing_indicators",
            columns=positive_columns,
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=20,
        ),
        Variant(
            name="top_450_raw_no_missing_indicators",
            columns=top_gain_columns(source_gain, 450),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=20,
        ),
        Variant(
            name="top_300_raw_no_missing_indicators",
            columns=top_gain_columns(source_gain, 300),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=20,
        ),
        Variant(
            name="domain_compact_350_no_missing_indicators",
            columns=domain_compact_columns(source_gain, 350),
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=20,
        ),
    ]

    rows = []
    importance_frames = []
    pruning_results_path = REPORTS_DIR / "relational_feature_pruning_results.csv"
    pruning_importance_path = (
        REPORTS_DIR / "relational_pruning_importance_by_experiment.csv"
    )
    if pruning_results_path.exists():
        pruning_results_path.unlink()
    if pruning_importance_path.exists():
        pruning_importance_path.unlink()

    for variant in variants:
        print(
            f"Training {variant.name}: {len(variant.columns)} raw columns, "
            f"missing indicators={variant.numeric_add_indicator}"
        )
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
        ).to_csv(pruning_results_path, index=False)
        pd.concat(importance_frames, ignore_index=True).to_csv(
            pruning_importance_path,
            index=False,
        )

        del importance, top_importance
        gc.collect()

    results = pd.DataFrame(rows).sort_values(
        ["validation_roc_auc", "validation_f1_class_1"],
        ascending=False,
    )
    results.to_csv(pruning_results_path, index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(
        pruning_importance_path,
        index=False,
    )

    best_name = str(results.iloc[0]["feature_set"])
    best_variant = next(variant for variant in variants if variant.name == best_name)
    best_threshold = float(results.iloc[0]["threshold"])

    print(f"Final training selected pruned feature set: {best_name}")
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

    test_probabilities = get_positive_probabilities(final_pipeline, X_test)
    final_metrics = pd.DataFrame(
        [
            {
                "feature_set": best_name,
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y_test, test_probabilities, 0.5),
            },
            {
                "feature_set": best_name,
                "threshold_strategy": "validation_selected",
                **evaluate_probabilities(y_test, test_probabilities, best_threshold),
            },
        ]
    )
    final_metrics.to_csv(
        REPORTS_DIR / "relational_pruned_final_test_metrics.csv",
        index=False,
    )

    selected_predictions = (test_probabilities >= best_threshold).astype(int)
    final_matrix = confusion_matrix(y_test, selected_predictions)
    pd.DataFrame(
        final_matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(REPORTS_DIR / "relational_pruned_final_confusion_matrix.csv")

    final_classification = classification_report(
        y_test,
        selected_predictions,
        zero_division=0,
    )
    (REPORTS_DIR / "relational_pruned_final_classification_report.txt").write_text(
        final_classification,
        encoding="utf-8",
    )

    final_importance = get_feature_importance(final_pipeline)
    final_importance.to_csv(
        REPORTS_DIR / "relational_pruned_final_feature_importance.csv",
        index=False,
    )
    write_report(
        results=results,
        final_metrics=final_metrics,
        final_confusion_matrix=final_matrix,
        final_classification_report=final_classification,
        top_importance=final_importance.head(40),
        source_gain=source_gain,
    )

    try:
        joblib.dump(final_pipeline, MODELS_DIR / "final_relational_pruned_lgbm.joblib")
    except OSError as error:
        print(f"Skipping model dump because the filesystem refused the write: {error}")

    print("Validation results")
    print(results.to_string(index=False))
    print("Final holdout metrics")
    print(final_metrics.to_string(index=False))
    print("Final confusion matrix")
    print(final_matrix)


if __name__ == "__main__":
    main()
