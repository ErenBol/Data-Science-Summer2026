from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
    as_markdown,
    build_lgbm_pipeline,
    top_gain_columns,
)
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
CATBOOST_REASONABLE_LIFT = 0.002
LIGHTGBM_OPTUNA_TRIALS = 25
CATBOOST_OPTUNA_TRIALS = 15


def build_reduced_matrix() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Index,
    pd.Index,
    pd.Index,
    pd.Index,
]:
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

    source_gain = pd.read_csv(REPORTS_DIR / "relational_source_column_gain.csv")
    reduced_columns = top_gain_columns(source_gain, 200)

    app_features = build_application_features(train_df)
    selected_app_columns = [
        column for column in reduced_columns if column in app_features.columns
    ]
    app_features = app_features[selected_app_columns]

    aligned_groups = {}
    for group_name in RELATIONAL_GROUPS:
        features = pd.read_pickle(PROCESSED_DIR / f"{group_name}_features.pkl")
        selected_group_columns = [
            column for column in reduced_columns if column in features.columns
        ]
        features = features[selected_group_columns]
        aligned_groups[group_name] = align_to_application(train_df, features)
        del features
        gc.collect()

    X_full = build_feature_matrix(app_features, aligned_groups, RELATIONAL_GROUPS)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_reduced = X_full[[column for column in reduced_columns if column in X_full.columns]].copy()

    return X_reduced, y, train_index, train_full_index, valid_index, test_index


def prepare_catboost_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = data.copy()
    categorical_columns = result.select_dtypes(exclude="number").columns.tolist()
    numeric_columns = result.select_dtypes(include="number").columns.tolist()

    for column in numeric_columns:
        result[column] = result[column].astype("float32")

    for column in categorical_columns:
        result[column] = result[column].astype("object").where(
            result[column].notna(),
            "__MISSING__",
        )
        result[column] = result[column].astype(str)

    return result, categorical_columns


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict | None = None,
) -> tuple[CatBoostClassifier, np.ndarray, int]:
    X_train_cb, categorical_columns = prepare_catboost_frame(X_train)
    X_valid_cb, _ = prepare_catboost_frame(X_valid)
    categorical_indices = [
        X_train_cb.columns.get_loc(column) for column in categorical_columns
    ]

    default_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "one_hot_max_size": 20,
        "max_ctr_complexity": 1,
        "ctr_leaf_count_limit": 64,
        "used_ram_limit": "3gb",
        "od_type": "Iter",
        "od_wait": 100,
    }
    if params is not None:
        default_params.update(params)

    model = CatBoostClassifier(**default_params)
    model.fit(
        Pool(X_train_cb, y_train, cat_features=categorical_indices),
        eval_set=Pool(X_valid_cb, y_valid, cat_features=categorical_indices),
        use_best_model=True,
    )
    probabilities = model.predict_proba(X_valid_cb)[:, 1]

    return model, probabilities, model.best_iteration_ or default_params["iterations"]


def evaluate_catboost_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    train_full_index: pd.Index,
    valid_index: pd.Index,
) -> tuple[dict, int]:
    model, probabilities, best_iteration = fit_catboost(
        X.loc[train_full_index],
        y.loc[train_full_index],
        X.loc[valid_index],
        y.loc[valid_index],
    )
    threshold_info = find_best_threshold(y.loc[valid_index], probabilities)
    metrics = evaluate_probabilities(
        y.loc[valid_index],
        probabilities,
        threshold_info["threshold"],
    )

    return (
        {
            "model": "catboost_baseline",
            "phase": "validation",
            "raw_feature_count": X.shape[1],
            "transformed_feature_count": X.shape[1],
            "threshold": threshold_info["threshold"],
            "best_iteration": best_iteration,
            **metrics,
        },
        best_iteration,
    )


def tune_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    train_full_index: pd.Index,
    valid_index: pd.Index,
) -> tuple[dict, dict, int]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 600, 1600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        }
        _, probabilities, _ = fit_catboost(
            X.loc[train_full_index],
            y.loc[train_full_index],
            X.loc[valid_index],
            y.loc[valid_index],
            params=params,
        )
        return roc_auc_score(y.loc[valid_index], probabilities)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=CATBOOST_OPTUNA_TRIALS, show_progress_bar=True)

    model, probabilities, best_iteration = fit_catboost(
        X.loc[train_full_index],
        y.loc[train_full_index],
        X.loc[valid_index],
        y.loc[valid_index],
        params=study.best_params,
    )
    threshold_info = find_best_threshold(y.loc[valid_index], probabilities)
    metrics = evaluate_probabilities(
        y.loc[valid_index],
        probabilities,
        threshold_info["threshold"],
    )

    return (
        {
            "model": "catboost_optuna",
            "phase": "validation",
            "raw_feature_count": X.shape[1],
            "transformed_feature_count": X.shape[1],
            "threshold": threshold_info["threshold"],
            "best_iteration": best_iteration,
            **metrics,
        },
        study.best_params,
        best_iteration,
    )


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    train_full_index: pd.Index,
    valid_index: pd.Index,
) -> tuple[dict, dict]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 180),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
            "importance_type": "gain",
        }
        pipeline = build_lgbm_pipeline(
            X_train=X.loc[train_full_index],
            numeric_add_indicator=False,
            one_hot_min_frequency=1000,
            one_hot_max_categories=15,
        )
        pipeline.named_steps["model"].set_params(**params)
        pipeline.fit(X.loc[train_full_index], y.loc[train_full_index])
        probabilities = get_positive_probabilities(pipeline, X.loc[valid_index])
        return roc_auc_score(y.loc[valid_index], probabilities)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=LIGHTGBM_OPTUNA_TRIALS, show_progress_bar=True)

    pipeline = build_lgbm_pipeline(
        X_train=X.loc[train_full_index],
        numeric_add_indicator=False,
        one_hot_min_frequency=1000,
        one_hot_max_categories=15,
    )
    pipeline.named_steps["model"].set_params(**study.best_params)
    pipeline.fit(X.loc[train_full_index], y.loc[train_full_index])
    probabilities = get_positive_probabilities(pipeline, X.loc[valid_index])
    threshold_info = find_best_threshold(y.loc[valid_index], probabilities)
    metrics = evaluate_probabilities(
        y.loc[valid_index],
        probabilities,
        threshold_info["threshold"],
    )

    return (
        {
            "model": "lightgbm_optuna",
            "phase": "validation",
            "raw_feature_count": X.shape[1],
            "transformed_feature_count": len(
                pipeline.named_steps["preprocessor"].get_feature_names_out()
            ),
            "threshold": threshold_info["threshold"],
            "best_iteration": study.best_params["n_estimators"],
            **metrics,
        },
        study.best_params,
    )


def fit_final_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    params: dict,
    iterations: int,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, str, CatBoostClassifier]:
    X_train, categorical_columns = prepare_catboost_frame(X.loc[train_index])
    X_test, _ = prepare_catboost_frame(X.loc[test_index])
    categorical_indices = [X_train.columns.get_loc(column) for column in categorical_columns]

    final_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": max(1, int(iterations)),
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
    }
    final_params.update(params)
    final_params["iterations"] = max(1, int(iterations))

    model = CatBoostClassifier(**final_params)
    model.fit(Pool(X_train, y.loc[train_index], cat_features=categorical_indices))
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "model": "catboost_final",
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y.loc[test_index], probabilities, 0.5),
            },
            {
                "model": "catboost_final",
                "threshold_strategy": "validation_selected",
                **evaluate_probabilities(y.loc[test_index], probabilities, threshold),
            },
        ]
    )
    predictions = (probabilities >= threshold).astype(int)

    return (
        metrics,
        confusion_matrix(y.loc[test_index], predictions),
        classification_report(y.loc[test_index], predictions, zero_division=0),
        model,
    )


def fit_final_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    params: dict,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, str, pd.DataFrame, object]:
    pipeline = build_lgbm_pipeline(
        X_train=X.loc[train_index],
        numeric_add_indicator=False,
        one_hot_min_frequency=1000,
        one_hot_max_categories=15,
    )
    pipeline.named_steps["model"].set_params(**params)
    pipeline.fit(X.loc[train_index], y.loc[train_index])
    probabilities = get_positive_probabilities(pipeline, X.loc[test_index])

    metrics = pd.DataFrame(
        [
            {
                "model": "lightgbm_optuna_final",
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y.loc[test_index], probabilities, 0.5),
            },
            {
                "model": "lightgbm_optuna_final",
                "threshold_strategy": "validation_selected",
                **evaluate_probabilities(y.loc[test_index], probabilities, threshold),
            },
        ]
    )
    predictions = (probabilities >= threshold).astype(int)

    return (
        metrics,
        confusion_matrix(y.loc[test_index], predictions),
        classification_report(y.loc[test_index], predictions, zero_division=0),
        get_feature_importance(pipeline),
        pipeline,
    )


def write_report(
    comparison: pd.DataFrame,
    final_metrics: pd.DataFrame,
    final_matrix: np.ndarray,
    final_classification: str,
    selected_path: str,
    selected_model_path: Path,
    best_params: dict,
    final_importance: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Reduced Feature Model Selection Report",
        "",
        "Run date: 2026-07-14",
        "",
        "## Decision Rule",
        "",
        f"CatBoost had to beat the reduced LightGBM validation ROC-AUC by at least `{CATBOOST_REASONABLE_LIFT}` to justify tuning CatBoost. Otherwise, Optuna tunes LightGBM on the same top-200 reduced feature set.",
        "",
        "## Validation Comparison",
        "",
        as_markdown(comparison),
        "",
        f"Selected path: `{selected_path}`",
        "",
        "## Final Holdout Metrics",
        "",
        as_markdown(final_metrics),
        "",
        "Confusion matrix at selected threshold:",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---:|---:|",
        f"| Actual 0 | {final_matrix[0, 0]:,} | {final_matrix[0, 1]:,} |",
        f"| Actual 1 | {final_matrix[1, 0]:,} | {final_matrix[1, 1]:,} |",
        "",
        "Classification report at selected threshold:",
        "",
        "```text",
        final_classification,
        "```",
        "",
        "## Best Parameters",
        "",
        "```python",
        repr(best_params),
        "```",
        "",
        "## Saved Model",
        "",
        f"`{selected_model_path}`",
    ]

    if final_importance is not None:
        lines.extend(
            [
                "",
                "## Final LightGBM Feature Importance",
                "",
                as_markdown(final_importance.head(50)),
            ]
        )

    (REPORTS_DIR / "reduced_model_selection_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, train_index, train_full_index, valid_index, test_index = build_reduced_matrix()
    previous_lgbm = pd.read_csv(REPORTS_DIR / "relational_feature_reduction_results.csv")
    previous_lgbm_row = previous_lgbm[
        previous_lgbm["feature_set"] == "top_200_no_missing_indicators"
    ].iloc[0]
    lgbm_validation_auc = float(previous_lgbm_row["validation_roc_auc"])

    rows = [
        {
            "model": "lightgbm_reduced_previous",
            "phase": "validation",
            "raw_feature_count": int(previous_lgbm_row["raw_feature_count"]),
            "transformed_feature_count": int(
                previous_lgbm_row["transformed_feature_count"]
            ),
            "threshold": float(previous_lgbm_row["threshold"]),
            "best_iteration": "",
            "roc_auc": lgbm_validation_auc,
            "average_precision": float(previous_lgbm_row["validation_average_precision"]),
            "accuracy": float(previous_lgbm_row["validation_accuracy"]),
            "precision_class_1": float(
                previous_lgbm_row["validation_precision_class_1"]
            ),
            "recall_class_1": float(previous_lgbm_row["validation_recall_class_1"]),
            "f1_class_1": float(previous_lgbm_row["validation_f1_class_1"]),
        }
    ]

    print("Training CatBoost baseline on reduced top-200 features")
    catboost_error = None
    try:
        catboost_row, catboost_best_iteration = evaluate_catboost_baseline(
            X,
            y,
            train_full_index,
            valid_index,
        )
        rows.append(catboost_row)
    except Exception as error:  # CatBoost can fail hard under tight RAM.
        catboost_error = repr(error)
        rows.append(
            {
                "model": "catboost_baseline",
                "phase": "validation",
                "raw_feature_count": X.shape[1],
                "transformed_feature_count": X.shape[1],
                "threshold": np.nan,
                "best_iteration": "",
                "roc_auc": np.nan,
                "average_precision": np.nan,
                "accuracy": np.nan,
                "precision_class_1": np.nan,
                "recall_class_1": np.nan,
                "f1_class_1": np.nan,
                "error": catboost_error,
            }
        )

    if (
        catboost_error is None
        and catboost_row["roc_auc"] >= lgbm_validation_auc + CATBOOST_REASONABLE_LIFT
    ):
        print("CatBoost improved enough; tuning CatBoost")
        selected_path = "catboost_optuna"
        tuned_row, best_params, best_iteration = tune_catboost(
            X,
            y,
            train_full_index,
            valid_index,
        )
        rows.append(tuned_row)
        final_metrics, matrix, final_classification, final_model = fit_final_catboost(
            X,
            y,
            train_index,
            test_index,
            best_params,
            best_iteration,
            tuned_row["threshold"],
        )
        model_path = MODELS_DIR / "final_reduced_catboost_optuna.cbm"
        final_model.save_model(model_path)
        final_importance = None
    else:
        print("CatBoost did not improve enough; tuning LightGBM with Optuna")
        selected_path = "lightgbm_optuna"
        tuned_row, best_params = tune_lightgbm(X, y, train_full_index, valid_index)
        rows.append(tuned_row)
        final_metrics, matrix, final_classification, final_importance, final_model = (
            fit_final_lightgbm(
                X,
                y,
                train_index,
                test_index,
                best_params,
                tuned_row["threshold"],
            )
        )
        model_path = MODELS_DIR / "final_reduced_lightgbm_optuna.joblib"
        joblib.dump(final_model, model_path, compress=3)
        final_importance.to_csv(
            REPORTS_DIR / "reduced_lightgbm_optuna_feature_importance.csv",
            index=False,
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS_DIR / "reduced_model_selection_validation.csv", index=False)
    final_metrics.to_csv(REPORTS_DIR / "reduced_model_selection_test_metrics.csv", index=False)
    pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(REPORTS_DIR / "reduced_model_selection_confusion_matrix.csv")
    (REPORTS_DIR / "reduced_model_selection_classification_report.txt").write_text(
        final_classification,
        encoding="utf-8",
    )

    write_report(
        comparison=comparison,
        final_metrics=final_metrics,
        final_matrix=matrix,
        final_classification=final_classification,
        selected_path=selected_path,
        selected_model_path=model_path,
        best_params=best_params,
        final_importance=final_importance,
    )

    print(comparison.to_string(index=False))
    print(final_metrics.to_string(index=False))
    print(f"Saved selected model to {model_path}")


if __name__ == "__main__":
    main()
