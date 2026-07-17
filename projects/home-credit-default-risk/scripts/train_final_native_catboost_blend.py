from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_blend import (
    CATBOOST_PARAMS,
    make_blend_probabilities,
    selected_transformed_matrices,
)
from scripts.run_advanced_relational_cv import (
    OPTUNA_LIGHTGBM_PARAMS,
    build_recent_domain_features,
    build_reduced_matrix,
    load_previous_te_rows,
)
from scripts.run_native_catboost_categorical import (
    build_native_catboost_matrix,
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
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_native_catboost_blend.joblib"
REPORT_PATH = REPORTS_DIR / "final_native_catboost_blend_report.md"
METRICS_PATH = REPORTS_DIR / "final_native_catboost_blend_holdout_metrics.csv"
CONFUSION_PATH = REPORTS_DIR / "final_native_catboost_blend_confusion_matrix.csv"
CLASSIFICATION_PATH = REPORTS_DIR / "final_native_catboost_blend_classification_report.txt"

LIGHTGBM_WEIGHT = 0.50
CATBOOST_WEIGHT = 0.50

BASELINE_HOLDOUT = {
    "model": "final_advanced_blend",
    "threshold": 0.6636775147709368,
    "roc_auc": 0.7953380230647751,
    "average_precision": 0.29816282212389184,
    "accuracy": 0.8626408467879616,
    "precision_class_1": 0.2845478164048002,
    "recall_class_1": 0.46324269889224573,
    "f1_class_1": 0.3525444512568976,
}


def fit_lightgbm_side(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[LGBMClassifier, object, np.ndarray, np.ndarray, np.ndarray]:
    X_train_selected, X_eval_selected, preprocessor, selected_mask, feature_names = (
        selected_transformed_matrices(X_train, y_train, X_eval)
    )
    model = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
    model.fit(X_train_selected, y_train)

    return model, preprocessor, selected_mask, feature_names, X_eval_selected


def fit_native_catboost_side(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series | None = None,
) -> tuple[CatBoostClassifier, pd.DataFrame, list[int], list[str], pd.DataFrame]:
    X_train_cb, categorical_indices, categorical_columns = prepare_native_catboost_frame(
        X_train
    )
    X_eval_cb, _, _ = prepare_native_catboost_frame(X_eval)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    fit_kwargs = {}
    if y_eval is not None:
        fit_kwargs["eval_set"] = Pool(
            X_eval_cb,
            y_eval,
            cat_features=categorical_indices,
        )
        fit_kwargs["use_best_model"] = True

    model.fit(
        Pool(X_train_cb, y_train, cat_features=categorical_indices),
        **fit_kwargs,
    )

    return model, X_train_cb, categorical_indices, categorical_columns, X_eval_cb


def write_report(
    metrics: pd.DataFrame,
    confusion: np.ndarray,
    classification: str,
    selected_threshold: float,
    transformed_feature_count: int,
    native_feature_count: int,
    native_categorical_count: int,
) -> None:
    selected_metrics = metrics[
        metrics["threshold_strategy"] == "validation_selected"
    ].iloc[0]
    comparison = pd.DataFrame(
        [
            BASELINE_HOLDOUT,
            {
                "model": "final_native_catboost_blend",
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
    delta = comparison.iloc[1].drop(labels="model") - comparison.iloc[0].drop(
        labels="model"
    )

    lines = [
        "# Final Native CatBoost Blend Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Setup",
        "",
        "- Same holdout split as `final_advanced_blend`: `test_size=0.2`, stratified, `random_state=42`.",
        "- Threshold selected on the same inner validation split from the training portion.",
        "- LightGBM side uses the advanced pruned transformed pipeline with fold-safe target encodings.",
        "- CatBoost side uses native raw categorical handling with `cat_features`; no `TE_*` target-encoded columns are passed to CatBoost.",
        "- Blend weights: `50/50` LightGBM/native CatBoost.",
        f"- LightGBM selected transformed features: `{transformed_feature_count}`",
        f"- Native CatBoost raw features: `{native_feature_count}`",
        f"- Native CatBoost categorical features: `{native_categorical_count}`",
        f"- Selected threshold: `{selected_threshold:.6f}`",
        "",
        "## Holdout Metrics",
        "",
        as_markdown(metrics),
        "",
        "## Comparison Against Previous Final Blend",
        "",
        as_markdown(comparison),
        "",
        "Delta, native blend minus previous final blend:",
        "",
        as_markdown(delta.rename("delta").reset_index().rename(columns={"index": "metric"})),
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
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Reading application data")
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

    train_index = train_index.to_numpy()
    test_index = test_index.to_numpy()
    train_full_index = train_full_index.to_numpy()
    valid_index = valid_index.to_numpy()

    print("Building feature matrices")
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()
    native_matrix = build_native_catboost_matrix(train_df)

    print("Selecting threshold on validation split")
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
    validation_probabilities = make_blend_probabilities(
        validation_lightgbm.predict_proba(X_lgbm_valid_selected)[:, 1],
        validation_catboost.predict_proba(X_native_valid)[:, 1],
        LIGHTGBM_WEIGHT,
    )
    threshold_info = find_best_threshold(y.loc[valid_index], validation_probabilities)
    selected_threshold = float(threshold_info["threshold"])

    print("Training final holdout blend")
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
    test_probabilities = make_blend_probabilities(
        final_lightgbm.predict_proba(X_lgbm_test_selected)[:, 1],
        final_catboost.predict_proba(X_native_test)[:, 1],
        LIGHTGBM_WEIGHT,
    )
    metrics = pd.DataFrame(
        [
            {
                "model": "final_native_catboost_blend",
                "threshold_strategy": "default_0.5",
                "lightgbm_weight": LIGHTGBM_WEIGHT,
                "catboost_weight": CATBOOST_WEIGHT,
                **evaluate_probabilities(y.loc[test_index], test_probabilities, 0.5),
            },
            {
                "model": "final_native_catboost_blend",
                "threshold_strategy": "validation_selected",
                "lightgbm_weight": LIGHTGBM_WEIGHT,
                "catboost_weight": CATBOOST_WEIGHT,
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

    metrics.to_csv(METRICS_PATH, index=False)
    pd.DataFrame(
        confusion,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(CONFUSION_PATH)
    CLASSIFICATION_PATH.write_text(classification, encoding="utf-8")

    selected_transformed_features = final_lightgbm_names[final_lightgbm_mask]
    joblib.dump(
        {
            "lightgbm_model": final_lightgbm,
            "lightgbm_preprocessor": final_lightgbm_preprocessor,
            "lightgbm_selected_mask": final_lightgbm_mask,
            "lightgbm_transformed_feature_names": final_lightgbm_names,
            "lightgbm_selected_transformed_features": selected_transformed_features,
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
            "lightgbm_weight": LIGHTGBM_WEIGHT,
            "catboost_weight": CATBOOST_WEIGHT,
            "lightgbm_params": OPTUNA_LIGHTGBM_PARAMS,
            "catboost_params": CATBOOST_PARAMS,
        },
        MODEL_PATH,
        compress=3,
    )

    write_report(
        metrics=metrics,
        confusion=confusion,
        classification=classification,
        selected_threshold=selected_threshold,
        transformed_feature_count=int(final_lightgbm_mask.sum()),
        native_feature_count=native_matrix.shape[1],
        native_categorical_count=len(native_categorical_columns),
    )

    print(metrics.to_string(index=False))
    print(confusion)
    print(f"Saved {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
