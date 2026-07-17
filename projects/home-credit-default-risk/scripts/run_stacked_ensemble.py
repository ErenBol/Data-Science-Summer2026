from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_blend import (
    make_blend_probabilities,
    selected_transformed_matrices,
)
from scripts.run_advanced_feature_pruning import build_advanced_fold_frames
from scripts.run_advanced_relational_cv import (
    N_SPLITS,
    OPTUNA_LIGHTGBM_PARAMS,
    build_recent_domain_features,
    build_reduced_matrix,
    load_previous_te_rows,
)
from scripts.run_native_catboost_categorical import (
    build_native_catboost_matrix,
    fit_native_catboost,
    prepare_native_catboost_frame,
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
from scripts.train_final_native_catboost_blend import (
    CATBOOST_PARAMS,
    fit_lightgbm_side,
    fit_native_catboost_side,
)
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_stacked_ensemble.joblib"
REPORT_PATH = REPORTS_DIR / "final_stacked_ensemble_report.md"
OOF_PREDICTIONS_PATH = REPORTS_DIR / "stacked_ensemble_oof_predictions.csv"
CV_RESULTS_PATH = REPORTS_DIR / "stacked_ensemble_cv_results.csv"
CV_SUMMARY_PATH = REPORTS_DIR / "stacked_ensemble_cv_summary.csv"
TOP_FEATURES_PATH = REPORTS_DIR / "stacked_ensemble_top_raw_features.csv"
HOLDOUT_METRICS_PATH = REPORTS_DIR / "final_stacked_ensemble_holdout_metrics.csv"
CONFUSION_PATH = REPORTS_DIR / "final_stacked_ensemble_confusion_matrix.csv"
CLASSIFICATION_PATH = REPORTS_DIR / "final_stacked_ensemble_classification_report.txt"

CURRENT_BEST_BLEND_CV_ROC_AUC = 0.7945306964243702
CURRENT_BEST_BLEND_CV_AVERAGE_PRECISION = 0.29086421064864393
CURRENT_NATIVE_BLEND_HOLDOUT = {
    "model": "final_native_catboost_blend",
    "threshold": 0.6689777135251893,
    "roc_auc": 0.796126495429448,
    "average_precision": 0.2990735301028307,
    "accuracy": 0.864722046079053,
    "precision_class_1": 0.28600586809542033,
    "recall_class_1": 0.4515609264853978,
    "f1_class_1": 0.35020306154326775,
}


def select_top_raw_features(reduced: pd.DataFrame, top_n: int = 10) -> list[str]:
    source_gain = pd.read_csv(REPORTS_DIR / "relational_source_column_gain.csv")
    numeric_columns = set(reduced.select_dtypes(include="number").columns)
    selected = []

    for column in source_gain["source_column"]:
        if column in numeric_columns and column not in selected:
            selected.append(column)
        if len(selected) == top_n:
            break

    if len(selected) < top_n:
        for column in reduced.select_dtypes(include="number").columns:
            if column not in selected:
                selected.append(column)
            if len(selected) == top_n:
                break

    return selected


def make_meta_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_meta_features(
    reduced: pd.DataFrame,
    lgbm_predictions: pd.Series,
    catboost_predictions: pd.Series,
    top_features: list[str],
) -> pd.DataFrame:
    meta = pd.DataFrame(index=reduced.index)
    meta["lgbm_oof_pred"] = lgbm_predictions.loc[reduced.index]
    meta["catboost_oof_pred"] = catboost_predictions.loc[reduced.index]
    for feature in top_features:
        meta[feature] = reduced[feature]

    return meta


def generate_oof_base_predictions(
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    native_matrix: pd.DataFrame,
    train_df: pd.DataFrame,
    previous_te: pd.DataFrame,
    y: pd.Series,
    indices: np.ndarray | None = None,
    label: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if indices is None:
        split_frame = reduced
        split_y = y
    else:
        split_frame = reduced.loc[indices]
        split_y = y.loc[indices]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_rows = []
    fold_rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(split_frame, split_y), start=1):
        train_index = split_frame.index.to_numpy()[train_pos]
        valid_index = split_frame.index.to_numpy()[valid_pos]
        print(f"{label} fold {fold}: LightGBM base OOF")
        X_lgbm_train, X_lgbm_valid = build_advanced_fold_frames(
            reduced=reduced,
            recent_domain=recent_domain,
            train_df=train_df,
            previous_te=previous_te,
            y=y,
            train_index=train_index,
            valid_index=valid_index,
        )
        (
            X_lgbm_train_selected,
            X_lgbm_valid_selected,
            _,
            selected_mask,
            _,
        ) = selected_transformed_matrices(
            X_lgbm_train,
            y.loc[train_index],
            X_lgbm_valid,
        )
        lightgbm = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
        lightgbm.fit(X_lgbm_train_selected, y.loc[train_index])
        lgbm_probabilities = lightgbm.predict_proba(X_lgbm_valid_selected)[:, 1]

        print(f"{label} fold {fold}: native CatBoost base OOF")
        catboost_probabilities, best_iteration, categorical_columns = fit_native_catboost(
            native_matrix.loc[train_index],
            y.loc[train_index],
            native_matrix.loc[valid_index],
            y.loc[valid_index],
        )
        for row_index, lgbm_probability, catboost_probability in zip(
            valid_index,
            lgbm_probabilities,
            catboost_probabilities,
        ):
            oof_rows.append(
                {
                    "index": row_index,
                    "fold": fold,
                    "lgbm_oof_pred": lgbm_probability,
                    "catboost_oof_pred": catboost_probability,
                    "target": int(y.loc[row_index]),
                }
            )
        fold_rows.append(
            {
                "label": label,
                "fold": fold,
                "lightgbm_transformed_feature_count": int(selected_mask.sum()),
                "catboost_best_iteration": best_iteration,
                "catboost_categorical_feature_count": len(categorical_columns),
                "lgbm_roc_auc": roc_auc_score(y.loc[valid_index], lgbm_probabilities),
                "catboost_roc_auc": roc_auc_score(
                    y.loc[valid_index],
                    catboost_probabilities,
                ),
                "lgbm_average_precision": average_precision_score(
                    y.loc[valid_index],
                    lgbm_probabilities,
                ),
                "catboost_average_precision": average_precision_score(
                    y.loc[valid_index],
                    catboost_probabilities,
                ),
            }
        )
        del (
            X_lgbm_train,
            X_lgbm_valid,
            X_lgbm_train_selected,
            X_lgbm_valid_selected,
            lightgbm,
        )
        gc.collect()

    oof = pd.DataFrame(oof_rows).set_index("index").sort_index()
    fold_results = pd.DataFrame(fold_rows)

    return oof, fold_results


def evaluate_meta_cv(meta_features: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(meta_features, y), start=1):
        train_index = meta_features.index.to_numpy()[train_pos]
        valid_index = meta_features.index.to_numpy()[valid_pos]
        pipeline = make_meta_pipeline()
        pipeline.fit(meta_features.loc[train_index], y.loc[train_index])
        probabilities = pipeline.predict_proba(meta_features.loc[valid_index])[:, 1]
        rows.append(
            {
                "model": "stacked_logistic_meta",
                "fold": fold,
                "roc_auc": roc_auc_score(y.loc[valid_index], probabilities),
                "average_precision": average_precision_score(
                    y.loc[valid_index],
                    probabilities,
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_cv(meta_cv: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "current_50_50_weighted_blend",
                "mean_cv_roc_auc": CURRENT_BEST_BLEND_CV_ROC_AUC,
                "std_cv_roc_auc": np.nan,
                "mean_cv_average_precision": CURRENT_BEST_BLEND_CV_AVERAGE_PRECISION,
                "std_cv_average_precision": np.nan,
            },
            {
                "model": "stacked_logistic_meta",
                "mean_cv_roc_auc": meta_cv["roc_auc"].mean(),
                "std_cv_roc_auc": meta_cv["roc_auc"].std(ddof=0),
                "mean_cv_average_precision": meta_cv["average_precision"].mean(),
                "std_cv_average_precision": meta_cv["average_precision"].std(ddof=0),
            },
        ]
    )


def train_final_stacked_holdout(
    train_df: pd.DataFrame,
    y: pd.Series,
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    native_matrix: pd.DataFrame,
    previous_te: pd.DataFrame,
    top_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, str, float]:
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

    print("Final stacking: train-split OOF base predictions for meta-model")
    train_oof, train_base_cv = generate_oof_base_predictions(
        reduced=reduced,
        recent_domain=recent_domain,
        native_matrix=native_matrix,
        train_df=train_df,
        previous_te=previous_te,
        y=y,
        indices=train_index,
        label="train_meta",
    )
    train_lgbm_oof = train_oof["lgbm_oof_pred"]
    train_catboost_oof = train_oof["catboost_oof_pred"]
    meta_train = build_meta_features(
        reduced.loc[train_index],
        train_lgbm_oof,
        train_catboost_oof,
        top_features,
    )
    meta_model = make_meta_pipeline()
    meta_model.fit(meta_train, y.loc[train_index])

    print("Final stacking: validation threshold selection")
    X_lgbm_train_full, X_lgbm_valid, _ = build_model_frames(
        reduced=reduced,
        recent_domain=recent_domain,
        train_df=train_df,
        previous=previous_te,
        y=y,
        train_index=train_full_index,
        transform_index=valid_index,
        oof_train=True,
    )
    validation_lightgbm, _, _, _, X_lgbm_valid_selected = fit_lightgbm_side(
        X_lgbm_train_full,
        y.loc[train_full_index],
        X_lgbm_valid,
    )
    validation_catboost, _, _, _, X_native_valid = fit_native_catboost_side(
        native_matrix.loc[train_full_index],
        y.loc[train_full_index],
        native_matrix.loc[valid_index],
        y.loc[valid_index],
    )
    meta_valid = build_meta_features(
        reduced.loc[valid_index],
        pd.Series(
            validation_lightgbm.predict_proba(X_lgbm_valid_selected)[:, 1],
            index=valid_index,
        ),
        pd.Series(
            validation_catboost.predict_proba(X_native_valid)[:, 1],
            index=valid_index,
        ),
        top_features,
    )
    valid_probabilities = meta_model.predict_proba(meta_valid)[:, 1]
    threshold_info = find_best_threshold(y.loc[valid_index], valid_probabilities)
    selected_threshold = float(threshold_info["threshold"])

    print("Final stacking: final base models and holdout evaluation")
    X_lgbm_train_final, X_lgbm_test, target_encoding_mappings = build_model_frames(
        reduced=reduced,
        recent_domain=recent_domain,
        train_df=train_df,
        previous=previous_te,
        y=y,
        train_index=train_index,
        transform_index=test_index,
        oof_train=True,
    )
    (
        final_lightgbm,
        final_lightgbm_preprocessor,
        final_lightgbm_mask,
        final_lightgbm_names,
        X_lgbm_test_selected,
    ) = fit_lightgbm_side(
        X_lgbm_train_final,
        y.loc[train_index],
        X_lgbm_test,
    )
    (
        final_catboost,
        _X_native_train_cb,
        native_categorical_indices,
        native_categorical_columns,
        X_native_test,
    ) = fit_native_catboost_side(
        native_matrix.loc[train_index],
        y.loc[train_index],
        native_matrix.loc[test_index],
    )
    meta_test = build_meta_features(
        reduced.loc[test_index],
        pd.Series(final_lightgbm.predict_proba(X_lgbm_test_selected)[:, 1], index=test_index),
        pd.Series(final_catboost.predict_proba(X_native_test)[:, 1], index=test_index),
        top_features,
    )
    test_probabilities = meta_model.predict_proba(meta_test)[:, 1]
    metrics = pd.DataFrame(
        [
            {
                "model": "final_stacked_ensemble",
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y.loc[test_index], test_probabilities, 0.5),
            },
            {
                "model": "final_stacked_ensemble",
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
    metrics.to_csv(HOLDOUT_METRICS_PATH, index=False)
    pd.DataFrame(
        confusion,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(CONFUSION_PATH)
    CLASSIFICATION_PATH.write_text(classification, encoding="utf-8")

    joblib.dump(
        {
            "meta_model": meta_model,
            "meta_features": meta_train.columns.tolist(),
            "top_raw_features": top_features,
            "lightgbm_model": final_lightgbm,
            "lightgbm_preprocessor": final_lightgbm_preprocessor,
            "lightgbm_selected_mask": final_lightgbm_mask,
            "lightgbm_transformed_feature_names": final_lightgbm_names,
            "lightgbm_selected_transformed_features": final_lightgbm_names[
                final_lightgbm_mask
            ],
            "lightgbm_raw_feature_columns": X_lgbm_train_final.columns.tolist(),
            "lightgbm_target_encoding_mappings": target_encoding_mappings,
            "lightgbm_dropped_transformed_features": sorted(
                DROPPED_TRANSFORMED_FEATURES
            ),
            "native_catboost_model": final_catboost,
            "native_catboost_feature_columns": native_matrix.columns.tolist(),
            "native_catboost_categorical_columns": native_categorical_columns,
            "native_catboost_categorical_indices": native_categorical_indices,
            "threshold": selected_threshold,
            "lightgbm_params": OPTUNA_LIGHTGBM_PARAMS,
            "catboost_params": CATBOOST_PARAMS,
            "train_meta_base_cv": train_base_cv,
        },
        MODEL_PATH,
        compress=3,
    )

    return metrics, confusion, classification, selected_threshold


def write_report(
    top_features: list[str],
    base_cv: pd.DataFrame,
    meta_cv: pd.DataFrame,
    cv_summary: pd.DataFrame,
    holdout_metrics: pd.DataFrame | None,
    confusion: np.ndarray | None,
    classification: str | None,
    selected_threshold: float | None,
) -> None:
    stacked_row = cv_summary[cv_summary["model"] == "stacked_logistic_meta"].iloc[0]
    cv_delta = pd.DataFrame(
        [
            {
                "metric": "mean_cv_roc_auc",
                "delta": stacked_row["mean_cv_roc_auc"] - CURRENT_BEST_BLEND_CV_ROC_AUC,
            },
            {
                "metric": "mean_cv_average_precision",
                "delta": stacked_row["mean_cv_average_precision"]
                - CURRENT_BEST_BLEND_CV_AVERAGE_PRECISION,
            },
        ]
    )
    lines = [
        "# Final Stacked Ensemble Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Setup",
        "",
        "- Base models: advanced-pruned LightGBM and native-categorical CatBoost.",
        "- Base-model out-of-fold predictions generated with the same 3-fold stratified splits as `run_advanced_relational_cv.py`.",
        "- Meta-model: logistic regression on `lgbm_oof_pred`, `catboost_oof_pred`, plus top raw gain features.",
        f"- Current weighted-blend CV baseline: ROC-AUC `{CURRENT_BEST_BLEND_CV_ROC_AUC:.6f}`, AP `{CURRENT_BEST_BLEND_CV_AVERAGE_PRECISION:.6f}`.",
        "",
        "## Top Raw Features Used By Meta-Model",
        "",
        as_markdown(pd.DataFrame({"feature": top_features})),
        "",
        "## Base Model OOF CV Diagnostics",
        "",
        as_markdown(base_cv),
        "",
        "## Meta-Model CV Results",
        "",
        as_markdown(meta_cv),
        "",
        "## CV Summary",
        "",
        as_markdown(cv_summary),
        "",
        "Delta versus current weighted blend:",
        "",
        as_markdown(cv_delta),
        "",
    ]

    if holdout_metrics is None:
        lines.extend(
            [
                "The stacked model did not improve CV ROC-AUC over the current 50/50 weighted blend, so no final stacked holdout model was trained or saved.",
                "",
            ]
        )
    else:
        selected_metrics = holdout_metrics[
            holdout_metrics["threshold_strategy"] == "validation_selected"
        ].iloc[0]
        comparison = pd.DataFrame(
            [
                CURRENT_NATIVE_BLEND_HOLDOUT,
                {
                    "model": "final_stacked_ensemble",
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
        holdout_delta = comparison.iloc[1].drop(labels="model") - comparison.iloc[
            0
        ].drop(labels="model")
        lines.extend(
            [
                "The stacked model improved CV ROC-AUC, so it was trained and evaluated on the holdout split.",
                "",
                f"Selected holdout threshold: `{selected_threshold:.6f}`",
                "",
                "## Holdout Metrics",
                "",
                as_markdown(holdout_metrics),
                "",
                "## Holdout Comparison Against Current Native Blend",
                "",
                as_markdown(comparison),
                "",
                "Delta, stacked model minus current native blend:",
                "",
                as_markdown(
                    holdout_delta.rename("delta")
                    .reset_index()
                    .rename(columns={"index": "metric"})
                ),
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
                "Saved model bundle:",
                "",
                f"`{MODEL_PATH}`",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Reading application data")
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    print("Building feature matrices")
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()
    native_matrix = build_native_catboost_matrix(train_df)
    top_features = select_top_raw_features(reduced, top_n=10)
    pd.DataFrame({"feature": top_features}).to_csv(TOP_FEATURES_PATH, index=False)

    print("Generating base OOF predictions")
    oof_predictions, base_cv = generate_oof_base_predictions(
        reduced=reduced,
        recent_domain=recent_domain,
        native_matrix=native_matrix,
        train_df=train_df,
        previous_te=previous_te,
        y=y,
        label="stack_cv",
    )
    oof_predictions.to_csv(OOF_PREDICTIONS_PATH)
    base_cv.to_csv(REPORTS_DIR / "stacked_ensemble_base_cv_results.csv", index=False)

    meta_features = build_meta_features(
        reduced,
        oof_predictions["lgbm_oof_pred"],
        oof_predictions["catboost_oof_pred"],
        top_features,
    )
    print("Evaluating logistic meta-model CV")
    meta_cv = evaluate_meta_cv(meta_features, y)
    cv_summary = summarize_cv(meta_cv)
    meta_cv.to_csv(CV_RESULTS_PATH, index=False)
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False)

    holdout_metrics = None
    confusion = None
    classification = None
    selected_threshold = None
    stacked_auc = float(
        cv_summary.loc[
            cv_summary["model"] == "stacked_logistic_meta",
            "mean_cv_roc_auc",
        ].iloc[0]
    )
    if stacked_auc > CURRENT_BEST_BLEND_CV_ROC_AUC:
        print("Stacked model improved CV ROC-AUC; training final holdout stack")
        holdout_metrics, confusion, classification, selected_threshold = (
            train_final_stacked_holdout(
                train_df=train_df,
                y=y,
                reduced=reduced,
                recent_domain=recent_domain,
                native_matrix=native_matrix,
                previous_te=previous_te,
                top_features=top_features,
            )
        )
    else:
        print("Stacked model did not improve CV ROC-AUC; skipping final holdout stack")

    write_report(
        top_features=top_features,
        base_cv=base_cv,
        meta_cv=meta_cv,
        cv_summary=cv_summary,
        holdout_metrics=holdout_metrics,
        confusion=confusion,
        classification=classification,
        selected_threshold=selected_threshold,
    )

    print(meta_cv.to_string(index=False))
    print(cv_summary.to_string(index=False))
    if holdout_metrics is not None:
        print(holdout_metrics.to_string(index=False))
        print(confusion)
        print(f"Saved {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
