from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Callable

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

from src.features import create_features
from src.preprocessing import build_preprocessor
from src.relational_features import (
    build_bureau_features,
    build_credit_card_features,
    build_installments_features,
    build_pos_cash_features,
    build_previous_application_features,
)
from src.thresholding import (
    evaluate_probabilities,
    find_best_threshold,
    get_positive_probabilities,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

LIGHTGBM_PARAMS = {
    "subsample": 1.0,
    "reg_lambda": 0.0,
    "num_leaves": 31,
    "n_estimators": 600,
    "min_child_samples": 50,
    "learning_rate": 0.02,
    "colsample_bytree": 0.85,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
    "importance_type": "gain",
}

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

APP_RAW_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "ORGANIZATION_TYPE",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE",
    "COMMONAREA_MODE",
    "ELEVATORS_MODE",
    "ENTRANCES_MODE",
    "FLOORSMAX_MODE",
    "FLOORSMIN_MODE",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI",
    "ELEVATORS_MEDI",
    "ENTRANCES_MEDI",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI",
    "LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_MEDI",
    "TOTALAREA_MODE",
    "OWN_CAR_AGE",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
]


def build_lgbm_pipeline(X_train: pd.DataFrame) -> Pipeline:
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        one_hot_sparse_output=True,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**LIGHTGBM_PARAMS)),
        ]
    )


def load_or_build_features(
    name: str,
    builder: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(exist_ok=True)
    cache_path = PROCESSED_DIR / f"{name}_features.pkl"

    if cache_path.exists():
        print(f"Loading cached {name} features")
        return pd.read_pickle(cache_path)

    print(f"Building {name} features")
    features = builder()
    features = features.replace([np.inf, -np.inf], np.nan)
    features.to_pickle(cache_path)
    print(f"Cached {features.shape[1]} {name} features to {cache_path.name}")

    gc.collect()
    return features


def align_to_application(
    train_df: pd.DataFrame,
    relational_features: pd.DataFrame,
) -> pd.DataFrame:
    aligned = train_df[["SK_ID_CURR"]].merge(
        relational_features,
        left_on="SK_ID_CURR",
        right_index=True,
        how="left",
    )
    aligned = aligned.drop(columns=["SK_ID_CURR"])
    aligned.index = train_df.index

    return aligned


def build_application_features(train_df: pd.DataFrame) -> pd.DataFrame:
    existing_columns = [
        column for column in APP_RAW_COLUMNS if column in train_df.columns
    ]
    app_features = create_features(train_df[existing_columns].copy())
    app_features = app_features.replace([np.inf, -np.inf], np.nan)

    return app_features


def build_feature_matrix(
    app_features: pd.DataFrame,
    aligned_groups: dict[str, pd.DataFrame],
    group_names: list[str],
) -> pd.DataFrame:
    frames = [app_features]
    frames.extend(aligned_groups[group_name] for group_name in group_names)
    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    return X


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    transformed_feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    gain_importance = pipeline.named_steps["model"].feature_importances_

    importance = pd.DataFrame(
        {
            "transformed_feature": transformed_feature_names,
            "gain_importance": gain_importance,
        }
    ).sort_values("gain_importance", ascending=False)

    return importance.reset_index(drop=True)


def evaluate_feature_set(
    name: str,
    X_full: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    train_full_index: pd.Index,
    valid_index: pd.Index,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    X_train = X_full.loc[train_full_index]
    X_valid = X_full.loc[valid_index]

    pipeline = build_lgbm_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    valid_probabilities = get_positive_probabilities(pipeline, X_valid)
    threshold_info = find_best_threshold(y_valid, valid_probabilities)
    metrics = evaluate_probabilities(
        y_true=y_valid,
        probabilities=valid_probabilities,
        threshold=threshold_info["threshold"],
    )

    importance = get_feature_importance(pipeline)
    top_features = importance.head(20)["transformed_feature"].tolist()

    result = {
        "feature_set": name,
        "raw_feature_count": X_full.shape[1],
        "threshold": threshold_info["threshold"],
        "validation_roc_auc": metrics["roc_auc"],
        "validation_average_precision": metrics["average_precision"],
        "validation_accuracy": metrics["accuracy"],
        "validation_precision_class_1": metrics["precision_class_1"],
        "validation_recall_class_1": metrics["recall_class_1"],
        "validation_f1_class_1": metrics["f1_class_1"],
        "top_20_features": "; ".join(top_features),
    }

    return result, importance


def summarize_profile_json() -> str:
    json_files = sorted(REPORTS_DIR.glob("*_profile.json"))

    if not json_files:
        return (
            "# Relational Profile JSON Summary\n\n"
            "No profile JSON files were found. Run "
            "`scripts/generate_relational_profiles.py` first.\n"
        )

    lines = ["# Relational Profile JSON Summary", ""]

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as file:
            profile = json.load(file)

        variables = profile.get("variables", {})
        rows = []
        for variable, metadata in variables.items():
            p_missing = metadata.get("p_missing")
            p_distinct = metadata.get("p_distinct")
            rows.append(
                {
                    "variable": variable,
                    "type": metadata.get("type", ""),
                    "p_missing": p_missing if p_missing is not None else 0,
                    "p_distinct": p_distinct if p_distinct is not None else 0,
                    "n_distinct": metadata.get("n_distinct", 0),
                }
            )

        summary = pd.DataFrame(rows)
        lines.append(f"## {json_path.stem.replace('_profile', '')}")

        if summary.empty:
            lines.extend(["", "No variables found.", ""])
            continue

        missing = summary.sort_values("p_missing", ascending=False).head(8)
        distinct = summary.sort_values("p_distinct", ascending=False).head(8)

        lines.append("")
        lines.append("Highest missingness fields:")
        for row in missing.itertuples(index=False):
            lines.append(
                f"- `{row.variable}`: missing {row.p_missing:.3f}, type {row.type}"
            )

        lines.append("")
        lines.append("Highest distinct-rate fields:")
        for row in distinct.itertuples(index=False):
            lines.append(
                f"- `{row.variable}`: distinct-rate {row.p_distinct:.3f}, "
                f"n_distinct {row.n_distinct}"
            )
        lines.append("")

    lines.extend(
        [
            "## Feature Hypotheses From Profile JSON",
            "",
            "- Bureau tables should add useful credit-history quantity, recency, "
            "active/closed status, overdue, debt-to-credit, and bureau-balance "
            "delinquency-status signals.",
            "- Previous applications should add approval/refusal history, previous "
            "amount ratios, product/channel mix, and decision-recency signals.",
            "- Installments should add direct repayment behavior: late-payment rate, "
            "underpayment rate, payment delay, and payment-to-installment ratios.",
            "- POS cash should add active/completed contract status, remaining "
            "installments, delinquency days past due, and completion progress.",
            "- Credit card balance should add utilization, payment-to-minimum, "
            "drawing behavior, receivables, and DPD signals.",
        ]
    )

    return "\n".join(lines) + "\n"


def write_report(
    results_df: pd.DataFrame,
    final_metrics_df: pd.DataFrame,
    final_confusion_matrix: np.ndarray,
    final_classification_report: str,
    best_feature_set: str,
    best_groups: list[str],
    top_importance: pd.DataFrame,
) -> None:
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

    lines = [
        "# Home Credit Relational Feature Experiments Report",
        "",
        "Run date: 2026-07-13",
        "",
        "## Setup",
        "",
        "- Base model: notebook 09 expanded application feature set.",
        "- Model: LightGBM with the same hyperparameters as the last selected setup.",
        "- Split: same random state, 20% holdout test, then 20% validation from the training split.",
        "- Threshold rule: same validation max-F1 threshold selection used in the previous notebooks.",
        "- Relational tables were aggregated to one row per `SK_ID_CURR` before joining.",
        "",
        "## Validation Results",
        "",
        as_markdown(results_df.drop(columns=["top_20_features"])),
        "",
        "## Selected Configuration",
        "",
        f"Best validation ROC-AUC feature set: `{best_feature_set}`",
        "",
        f"Included relational groups: `{', '.join(best_groups) if best_groups else 'none'}`",
        "",
        "## Holdout Test Metrics",
        "",
        as_markdown(final_metrics_df),
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
        final_classification_report,
        "```",
        "",
        "## Top Final Feature Importances",
        "",
        as_markdown(top_importance),
        "",
        "## Interpretation",
        "",
        "The final model still uses the application-table engineered signals, but it now also learns "
        "from historical credit behavior summarized across the secondary Home Credit tables. The "
        "most important relational signals should be interpreted as customer-level aggregates, not "
        "individual loan records.",
        "",
        "If the result remains below 0.80 ROC-AUC, the next performance step is not more threshold "
        "work. Thresholds change precision/recall tradeoffs, not ranking quality. The next step is "
        "LightGBM tuning on the joined relational feature matrix, plus more targeted aggregate "
        "features around recent history windows and category-specific bureau/previous-loan behavior.",
    ]

    (REPORTS_DIR / "relational_feature_experiments_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    profile_summary = summarize_profile_json()
    (REPORTS_DIR / "relational_profile_json_summary.md").write_text(
        profile_summary,
        encoding="utf-8",
    )

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

    print("Building application features")
    app_features = build_application_features(train_df)

    builders: dict[str, Callable[[], pd.DataFrame]] = {
        "bureau": lambda: build_bureau_features(
            RAW_DATA_DIR / "bureau.csv",
            RAW_DATA_DIR / "bureau_balance.csv",
        ),
        "previous_application": lambda: build_previous_application_features(
            RAW_DATA_DIR / "previous_application.csv"
        ),
        "installments_payments": lambda: build_installments_features(
            RAW_DATA_DIR / "installments_payments.csv"
        ),
        "POS_CASH_balance": lambda: build_pos_cash_features(
            RAW_DATA_DIR / "POS_CASH_balance.csv"
        ),
        "credit_card_balance": lambda: build_credit_card_features(
            RAW_DATA_DIR / "credit_card_balance.csv"
        ),
    }

    aligned_groups = {}
    for group_name, builder in builders.items():
        features = load_or_build_features(group_name, builder)
        aligned_groups[group_name] = align_to_application(train_df, features)
        print(f"Aligned {group_name}: {aligned_groups[group_name].shape}")
        del features
        gc.collect()

    experiment_specs: list[tuple[str, list[str]]] = [
        ("app_expanded_only", []),
    ]
    experiment_specs.extend(
        (f"app_plus_{group_name}", [group_name])
        for group_name in builders
    )

    cumulative_groups: list[str] = []
    for group_name in builders:
        cumulative_groups.append(group_name)
        cumulative_name = "app_plus_" + "_plus_".join(cumulative_groups)
        if cumulative_name not in {name for name, _ in experiment_specs}:
            experiment_specs.append((cumulative_name, cumulative_groups.copy()))

    results = []
    importance_rows = []

    for feature_set_name, group_names in experiment_specs:
        print(f"Training {feature_set_name}")
        X_full = build_feature_matrix(app_features, aligned_groups, group_names)
        result, importance = evaluate_feature_set(
            name=feature_set_name,
            X_full=X_full,
            y_train=y_train,
            y_valid=y_valid,
            train_full_index=train_full_index,
            valid_index=valid_index,
        )
        result["groups"] = ",".join(group_names)
        results.append(result)

        top_importance = importance.head(50).copy()
        top_importance.insert(0, "feature_set", feature_set_name)
        importance_rows.append(top_importance)

        del X_full, importance, top_importance
        gc.collect()

    results_df = pd.DataFrame(results).sort_values(
        ["validation_roc_auc", "validation_f1_class_1"],
        ascending=False,
    )
    results_df.to_csv(
        REPORTS_DIR / "relational_feature_experiment_results.csv",
        index=False,
    )

    importance_df = pd.concat(importance_rows, ignore_index=True)
    importance_df.to_csv(
        REPORTS_DIR / "relational_feature_importance_by_experiment.csv",
        index=False,
    )

    best_row = results_df.iloc[0]
    best_feature_set = str(best_row["feature_set"])
    best_groups = str(best_row["groups"]).split(",") if best_row["groups"] else []
    best_threshold = float(best_row["threshold"])

    print(f"Final training selected feature set: {best_feature_set}")
    X_best = build_feature_matrix(app_features, aligned_groups, best_groups)
    X_train_full = X_best.loc[train_index]
    X_test = X_best.loc[test_index]

    final_pipeline = build_lgbm_pipeline(X_train_full)
    final_pipeline.fit(X_train_full, y_train_full)

    test_probabilities = get_positive_probabilities(final_pipeline, X_test)
    final_default_metrics = evaluate_probabilities(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=0.5,
    )
    final_selected_metrics = evaluate_probabilities(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=best_threshold,
    )

    final_metrics_df = pd.DataFrame(
        [
            {
                "feature_set": best_feature_set,
                "threshold_strategy": "default_0.5",
                **final_default_metrics,
            },
            {
                "feature_set": best_feature_set,
                "threshold_strategy": "validation_selected",
                **final_selected_metrics,
            },
        ]
    )
    final_metrics_df.to_csv(
        REPORTS_DIR / "relational_final_test_metrics.csv",
        index=False,
    )

    selected_predictions = (test_probabilities >= best_threshold).astype(int)
    final_matrix = confusion_matrix(y_test, selected_predictions)
    final_classification = classification_report(
        y_test,
        selected_predictions,
        zero_division=0,
    )
    pd.DataFrame(
        final_matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(REPORTS_DIR / "relational_final_confusion_matrix.csv")
    (REPORTS_DIR / "relational_final_classification_report.txt").write_text(
        final_classification,
        encoding="utf-8",
    )

    final_importance = get_feature_importance(final_pipeline)
    final_importance.to_csv(
        REPORTS_DIR / "relational_final_feature_importance.csv",
        index=False,
    )

    joblib.dump(final_pipeline, MODELS_DIR / "final_relational_lgbm.joblib")

    write_report(
        results_df=results_df,
        final_metrics_df=final_metrics_df,
        final_confusion_matrix=final_matrix,
        final_classification_report=final_classification,
        best_feature_set=best_feature_set,
        best_groups=best_groups,
        top_importance=final_importance.head(40),
    )

    print("Validation results")
    print(results_df.drop(columns=["top_20_features"]).to_string(index=False))
    print("Final holdout metrics")
    print(final_metrics_df.to_string(index=False))
    print("Final confusion matrix")
    print(final_matrix)


if __name__ == "__main__":
    main()
