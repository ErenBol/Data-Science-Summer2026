from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_feature_pruning import (
    build_advanced_fold_frames,
    make_preprocessor,
)
from scripts.run_advanced_relational_cv import (
    N_SPLITS,
    build_recent_domain_features,
    build_reduced_matrix,
    load_previous_te_rows,
)
from scripts.run_relational_feature_experiments import (
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from scripts.run_relational_feature_pruning import as_markdown
from scripts.train_final_advanced_pruned_holdout import (
    DROPPED_TRANSFORMED_FEATURES,
    build_model_frames,
)
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_advanced_pruned_lgbm_optuna.joblib"
REPORT_PATH = REPORTS_DIR / "final_advanced_pruned_optuna_report.md"
TRIALS_PATH = REPORTS_DIR / "final_advanced_pruned_optuna_trials.csv"
CV_RESULTS_PATH = REPORTS_DIR / "final_advanced_pruned_optuna_cv_results.csv"
HOLDOUT_METRICS_PATH = REPORTS_DIR / "final_advanced_pruned_optuna_holdout_metrics.csv"
CONFUSION_PATH = REPORTS_DIR / "final_advanced_pruned_optuna_confusion_matrix.csv"
CLASSIFICATION_PATH = (
    REPORTS_DIR / "final_advanced_pruned_optuna_classification_report.txt"
)
FEATURES_PATH = REPORTS_DIR / "final_advanced_pruned_optuna_transformed_features.csv"

N_OPTUNA_TRIALS = 35

BASELINE_SELECTED_METRICS = {
    "model": "final_advanced_pruned_lgbm",
    "threshold": 0.670437625096521,
    "roc_auc": 0.794353430966071,
    "average_precision": 0.2967048111538493,
    "accuracy": 0.867112173389916,
    "precision_class_1": 0.28939075630252103,
    "recall_class_1": 0.44390735146022153,
    "f1_class_1": 0.3503696049598601,
}


def lightgbm_params_from_trial(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 600, 1800),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 96),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 60, 260),
        "subsample": trial.suggest_float("subsample", 0.75, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
        "importance_type": "gain",
    }


def prepare_cv_folds(
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    train_df: pd.DataFrame,
    previous_te: pd.DataFrame,
    y: pd.Series,
) -> list[dict]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
        train_index = reduced.index.to_numpy()[train_pos]
        valid_index = reduced.index.to_numpy()[valid_pos]
        print(f"Preparing transformed matrices for CV fold {fold}")
        X_train, X_valid = build_advanced_fold_frames(
            reduced=reduced,
            recent_domain=recent_domain,
            train_df=train_df,
            previous_te=previous_te,
            y=y,
            train_index=train_index,
            valid_index=valid_index,
        )
        preprocessor = make_preprocessor(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train, y.loc[train_index])
        X_valid_transformed = preprocessor.transform(X_valid)
        transformed_names = np.asarray(preprocessor.get_feature_names_out())
        selected_mask = np.asarray(
            [name not in DROPPED_TRANSFORMED_FEATURES for name in transformed_names]
        )
        folds.append(
            {
                "fold": fold,
                "X_train": X_train_transformed[:, selected_mask],
                "X_valid": X_valid_transformed[:, selected_mask],
                "y_train": y.loc[train_index].to_numpy(),
                "y_valid": y.loc[valid_index].to_numpy(),
                "transformed_feature_count": int(selected_mask.sum()),
            }
        )
        del X_train, X_valid, X_train_transformed, X_valid_transformed
        gc.collect()

    return folds


def evaluate_params_cv(params: dict, folds: list[dict]) -> pd.DataFrame:
    rows = []
    for fold_data in folds:
        model = LGBMClassifier(**params)
        model.fit(fold_data["X_train"], fold_data["y_train"])
        probabilities = model.predict_proba(fold_data["X_valid"])[:, 1]
        rows.append(
            {
                "fold": fold_data["fold"],
                "transformed_feature_count": fold_data["transformed_feature_count"],
                "roc_auc": roc_auc_score(fold_data["y_valid"], probabilities),
                "average_precision": average_precision_score(
                    fold_data["y_valid"],
                    probabilities,
                ),
            }
        )
        del model
        gc.collect()

    return pd.DataFrame(rows)


def run_optuna(folds: list[dict]) -> tuple[optuna.Study, pd.DataFrame]:
    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = lightgbm_params_from_trial(trial)
        fold_results = evaluate_params_cv(params, folds)
        mean_roc_auc = float(fold_results["roc_auc"].mean())
        mean_ap = float(fold_results["average_precision"].mean())
        trial.set_user_attr("mean_average_precision", mean_ap)
        trial.set_user_attr("fold_roc_auc", fold_results["roc_auc"].tolist())
        trial.set_user_attr("fold_average_precision", fold_results["average_precision"].tolist())
        trial_rows.append(
            {
                "trial": trial.number,
                "mean_roc_auc": mean_roc_auc,
                "mean_average_precision": mean_ap,
                **params,
            }
        )
        pd.DataFrame(trial_rows).to_csv(TRIALS_PATH, index=False)

        return mean_roc_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    trials = pd.DataFrame(trial_rows).sort_values("mean_roc_auc", ascending=False)
    trials.to_csv(TRIALS_PATH, index=False)

    return study, trials


def fit_transformed_model_with_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    params: dict,
) -> tuple[LGBMClassifier, object, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = make_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_eval_transformed = preprocessor.transform(X_eval)
    transformed_names = np.asarray(preprocessor.get_feature_names_out())
    selected_mask = np.asarray(
        [name not in DROPPED_TRANSFORMED_FEATURES for name in transformed_names]
    )
    model = LGBMClassifier(**params)
    model.fit(X_train_transformed[:, selected_mask], y_train)

    return (
        model,
        preprocessor,
        selected_mask,
        transformed_names,
        X_eval_transformed[:, selected_mask],
    )


def train_final_holdout(
    train_df: pd.DataFrame,
    y: pd.Series,
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    previous: pd.DataFrame,
    best_params: dict,
) -> tuple[pd.DataFrame, np.ndarray, str, float, dict, np.ndarray, np.ndarray, object, LGBMClassifier]:
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

    train_index = train_index.to_numpy()
    test_index = test_index.to_numpy()
    train_full_index = train_full_index.to_numpy()
    valid_index = valid_index.to_numpy()

    X_train_full, X_valid, _ = build_model_frames(
        reduced=reduced,
        recent_domain=recent_domain,
        train_df=train_df,
        previous=previous,
        y=y,
        train_index=train_full_index,
        transform_index=valid_index,
        oof_train=True,
    )
    validation_model, _, _, _, X_valid_selected = fit_transformed_model_with_params(
        X_train=X_train_full,
        y_train=y.loc[train_full_index],
        X_eval=X_valid,
        params=best_params,
    )
    valid_probabilities = validation_model.predict_proba(X_valid_selected)[:, 1]
    threshold_info = find_best_threshold(y.loc[valid_index], valid_probabilities)
    selected_threshold = float(threshold_info["threshold"])

    X_train_final, X_test, target_encoding_mappings = build_model_frames(
        reduced=reduced,
        recent_domain=recent_domain,
        train_df=train_df,
        previous=previous,
        y=y,
        train_index=train_index,
        transform_index=test_index,
        oof_train=True,
    )
    (
        final_model,
        final_preprocessor,
        final_mask,
        final_names,
        X_test_selected,
    ) = fit_transformed_model_with_params(
        X_train=X_train_final,
        y_train=y.loc[train_index],
        X_eval=X_test,
        params=best_params,
    )
    test_probabilities = final_model.predict_proba(X_test_selected)[:, 1]
    metrics = pd.DataFrame(
        [
            {
                "model": "final_advanced_pruned_lgbm_optuna",
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y.loc[test_index], test_probabilities, 0.5),
            },
            {
                "model": "final_advanced_pruned_lgbm_optuna",
                "threshold_strategy": "validation_selected",
                **evaluate_probabilities(
                    y.loc[test_index],
                    test_probabilities,
                    selected_threshold,
                ),
            },
        ]
    )
    selected_predictions = (test_probabilities >= selected_threshold).astype(int)
    confusion = confusion_matrix(y.loc[test_index], selected_predictions)
    classification = classification_report(
        y.loc[test_index],
        selected_predictions,
        zero_division=0,
    )

    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "selected_mask": final_mask,
            "transformed_feature_names": final_names,
            "selected_transformed_features": final_names[final_mask],
            "dropped_transformed_features": sorted(DROPPED_TRANSFORMED_FEATURES),
            "raw_feature_columns": X_train_final.columns.tolist(),
            "target_encoding_mappings": target_encoding_mappings,
            "threshold": selected_threshold,
            "lightgbm_params": best_params,
            "optuna_trials": N_OPTUNA_TRIALS,
        },
        MODEL_PATH,
        compress=3,
    )

    return (
        metrics,
        confusion,
        classification,
        selected_threshold,
        target_encoding_mappings,
        final_names,
        final_mask,
        final_preprocessor,
        final_model,
    )


def write_report(
    best_params: dict,
    best_cv_results: pd.DataFrame,
    trials: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    confusion: np.ndarray,
    classification: str,
    threshold: float,
    transformed_feature_count: int,
) -> None:
    selected_metrics = holdout_metrics[
        holdout_metrics["threshold_strategy"] == "validation_selected"
    ].iloc[0]
    comparison = pd.DataFrame(
        [
            BASELINE_SELECTED_METRICS,
            {
                "model": "final_advanced_pruned_lgbm_optuna",
                "threshold": selected_metrics["threshold"],
                "roc_auc": selected_metrics["roc_auc"],
                "average_precision": selected_metrics["average_precision"],
                "accuracy": selected_metrics["accuracy"],
                "precision_class_1": selected_metrics["precision_class_1"],
                "recall_class_1": selected_metrics["recall_class_1"],
                "f1_class_1": selected_metrics["f1_class_1"],
            },
        ]
    )
    deltas = comparison.iloc[1].drop(labels="model") - comparison.iloc[0].drop(
        labels="model"
    )
    cv_summary = pd.DataFrame(
        [
            {
                "mean_cv_roc_auc": best_cv_results["roc_auc"].mean(),
                "std_cv_roc_auc": best_cv_results["roc_auc"].std(ddof=0),
                "mean_cv_average_precision": best_cv_results[
                    "average_precision"
                ].mean(),
                "std_cv_average_precision": best_cv_results[
                    "average_precision"
                ].std(ddof=0),
                "mean_transformed_feature_count": best_cv_results[
                    "transformed_feature_count"
                ].mean(),
            }
        ]
    )
    params_df = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in best_params.items()]
    )

    lines = [
        "# Final Advanced Pruned Optuna Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Setup",
        "",
        f"- Optuna trials: `{N_OPTUNA_TRIALS}`",
        "- Objective: mean 3-fold stratified CV ROC-AUC.",
        "- Feature set: `advanced_drop_2_weakest_transformed`.",
        "- Same fold-safe target encoding and preprocessing setup as `run_advanced_relational_cv.py`.",
        "- Same holdout split and validation-threshold selection flow as `final_advanced_pruned_lgbm`.",
        f"- Final selected transformed feature count: `{transformed_feature_count}`",
        f"- Selected threshold: `{threshold:.6f}`",
        "",
        "## Best Parameters",
        "",
        as_markdown(params_df),
        "",
        "## Best CV Metrics",
        "",
        as_markdown(cv_summary),
        "",
        "## Per-Fold CV Metrics",
        "",
        as_markdown(best_cv_results),
        "",
        "## Holdout Metrics",
        "",
        as_markdown(holdout_metrics),
        "",
        "## Holdout Comparison",
        "",
        as_markdown(comparison),
        "",
        "Delta, tuned minus previous advanced pruned:",
        "",
        as_markdown(deltas.rename("delta").reset_index().rename(columns={"index": "metric"})),
        "",
        "Confusion matrix at selected threshold:",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---:|---:|",
        f"| Actual 0 | {confusion[0, 0]:,} | {confusion[0, 1]:,} |",
        f"| Actual 1 | {confusion[1, 0]:,} | {confusion[1, 1]:,} |",
        "",
        "Classification report at selected threshold:",
        "",
        "```text",
        classification,
        "```",
        "",
        "Top Optuna trials:",
        "",
        as_markdown(trials.head(10)),
        "",
        "Saved model bundle:",
        "",
        f"`{MODEL_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Reading application data")
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    print("Building advanced pruned feature inputs")
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()

    folds = prepare_cv_folds(
        reduced=reduced,
        recent_domain=recent_domain,
        train_df=train_df,
        previous_te=previous_te,
        y=y,
    )

    print(f"Running Optuna LightGBM tuning for {N_OPTUNA_TRIALS} trials")
    study, trials = run_optuna(folds)
    best_params = lightgbm_params_from_trial(study.best_trial)
    best_cv_results = evaluate_params_cv(best_params, folds)
    best_cv_results.to_csv(CV_RESULTS_PATH, index=False)

    print("Training final holdout model with best parameters")
    (
        holdout_metrics,
        confusion,
        classification,
        selected_threshold,
        _target_encoding_mappings,
        final_names,
        final_mask,
        _final_preprocessor,
        _final_model,
    ) = train_final_holdout(
        train_df=train_df,
        y=y,
        reduced=reduced,
        recent_domain=recent_domain,
        previous=previous_te,
        best_params=best_params,
    )
    holdout_metrics.to_csv(HOLDOUT_METRICS_PATH, index=False)
    pd.DataFrame(
        confusion,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(CONFUSION_PATH)
    CLASSIFICATION_PATH.write_text(classification, encoding="utf-8")
    pd.DataFrame({"transformed_feature": final_names[final_mask]}).to_csv(
        FEATURES_PATH,
        index=False,
    )

    write_report(
        best_params=best_params,
        best_cv_results=best_cv_results,
        trials=trials,
        holdout_metrics=holdout_metrics,
        confusion=confusion,
        classification=classification,
        threshold=selected_threshold,
        transformed_feature_count=int(final_mask.sum()),
    )

    print("Best parameters:")
    print(best_params)
    print(best_cv_results.to_string(index=False))
    print(holdout_metrics.to_string(index=False))
    print(confusion)
    print(f"Saved {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
