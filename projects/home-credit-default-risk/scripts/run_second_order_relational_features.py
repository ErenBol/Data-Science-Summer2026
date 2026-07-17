from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
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

from scripts.run_advanced_blend import (
    CATBOOST_PARAMS,
    fit_catboost,
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
from scripts.run_relational_feature_experiments import (
    PROCESSED_DIR,
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
from src.relational_features import build_second_order_recent_history_features
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_second_order_advanced_blend.joblib"
REPORT_PATH = REPORTS_DIR / "second_order_relational_features_report.md"
CV_RESULTS_PATH = REPORTS_DIR / "second_order_relational_features_cv_results.csv"
FEATURES_PATH = REPORTS_DIR / "second_order_relational_feature_columns.csv"
HOLDOUT_METRICS_PATH = REPORTS_DIR / "second_order_advanced_blend_holdout_metrics.csv"
CONFUSION_PATH = REPORTS_DIR / "second_order_advanced_blend_confusion_matrix.csv"
CLASSIFICATION_PATH = REPORTS_DIR / "second_order_advanced_blend_classification_report.txt"

BASELINE_CV_ROC_AUC = 0.7925488471724659
BASELINE_CV_AVERAGE_PRECISION = 0.2880674450643672
BASELINE_HOLDOUT_BLEND = {
    "model": "final_advanced_blend",
    "threshold": 0.6636775147709368,
    "roc_auc": 0.7953380230647751,
    "average_precision": 0.29816282212389184,
    "accuracy": 0.8626408467879616,
    "precision_class_1": 0.2845478164048002,
    "recall_class_1": 0.46324269889224573,
    "f1_class_1": 0.3525444512568976,
}
BLEND_LIGHTGBM_WEIGHT = 0.60


def align_to_application(train_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    aligned = train_df[["SK_ID_CURR"]].merge(
        features,
        left_on="SK_ID_CURR",
        right_index=True,
        how="left",
    )
    aligned = aligned.drop(columns=["SK_ID_CURR"])
    aligned.index = train_df.index

    return aligned


def build_second_order_aligned_features(train_df: pd.DataFrame) -> pd.DataFrame:
    cache_path = PROCESSED_DIR / "second_order_recent_history_features_v1.pkl"
    if cache_path.exists():
        features = pd.read_pickle(cache_path)
    else:
        features = build_second_order_recent_history_features(RAW_DATA_DIR)
        features.to_pickle(cache_path)

    aligned = align_to_application(train_df, features)
    pd.DataFrame({"feature": aligned.columns}).to_csv(FEATURES_PATH, index=False)

    return aligned


def build_augmented_recent_features(train_df: pd.DataFrame) -> pd.DataFrame:
    recent_domain = build_recent_domain_features(train_df)
    second_order = build_second_order_aligned_features(train_df)
    combined = pd.concat([recent_domain, second_order], axis=1)

    return combined.replace([np.inf, -np.inf], np.nan)


def run_lightgbm_cv(
    reduced: pd.DataFrame,
    augmented_recent: pd.DataFrame,
    train_df: pd.DataFrame,
    previous_te: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
        train_index = reduced.index.to_numpy()[train_pos]
        valid_index = reduced.index.to_numpy()[valid_pos]
        print(f"Fold {fold}: building second-order advanced_drop_2 matrices")
        X_train, X_valid = build_advanced_fold_frames(
            reduced=reduced,
            recent_domain=augmented_recent,
            train_df=train_df,
            previous_te=previous_te,
            y=y,
            train_index=train_index,
            valid_index=valid_index,
        )
        X_train_selected, X_valid_selected, _, selected_mask, _ = (
            selected_transformed_matrices(
                X_train,
                y.loc[train_index],
                X_valid,
            )
        )
        model = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
        model.fit(X_train_selected, y.loc[train_index])
        probabilities = model.predict_proba(X_valid_selected)[:, 1]
        rows.append(
            {
                "feature_set": "advanced_drop_2_plus_second_order",
                "fold": fold,
                "raw_feature_count": X_train.shape[1],
                "transformed_feature_count": int(selected_mask.sum()),
                "roc_auc": roc_auc_score(y.loc[valid_index], probabilities),
                "average_precision": average_precision_score(
                    y.loc[valid_index],
                    probabilities,
                ),
            }
        )
        del X_train, X_valid, X_train_selected, X_valid_selected, model
        gc.collect()

    return pd.DataFrame(rows)


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": "advanced_drop_2_weakest_transformed_baseline",
                "mean_cv_roc_auc": BASELINE_CV_ROC_AUC,
                "std_cv_roc_auc": np.nan,
                "mean_cv_average_precision": BASELINE_CV_AVERAGE_PRECISION,
                "std_cv_average_precision": np.nan,
                "mean_transformed_feature_count": 363.0,
            },
            {
                "feature_set": "advanced_drop_2_plus_second_order",
                "mean_cv_roc_auc": cv_results["roc_auc"].mean(),
                "std_cv_roc_auc": cv_results["roc_auc"].std(ddof=0),
                "mean_cv_average_precision": cv_results["average_precision"].mean(),
                "std_cv_average_precision": cv_results["average_precision"].std(ddof=0),
                "mean_transformed_feature_count": cv_results[
                    "transformed_feature_count"
                ].mean(),
            },
        ]
    )


def train_second_order_blend(
    train_df: pd.DataFrame,
    y: pd.Series,
    reduced: pd.DataFrame,
    augmented_recent: pd.DataFrame,
    previous_te: pd.DataFrame,
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

    print("Second-order blend: selecting threshold on validation split")
    X_train_full, X_valid, _ = build_model_frames(
        reduced=reduced,
        recent_domain=augmented_recent,
        train_df=train_df,
        previous=previous_te,
        y=y,
        train_index=train_full_index,
        transform_index=valid_index,
        oof_train=True,
    )
    X_train_full_selected, X_valid_selected, _, _, _ = selected_transformed_matrices(
        X_train_full,
        y.loc[train_full_index],
        X_valid,
    )
    validation_lightgbm = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
    validation_lightgbm.fit(X_train_full_selected, y.loc[train_full_index])
    validation_catboost = fit_catboost(
        X_train_full_selected,
        y.loc[train_full_index].to_numpy(),
        X_valid_selected,
        y.loc[valid_index].to_numpy(),
    )
    validation_probabilities = make_blend_probabilities(
        validation_lightgbm.predict_proba(X_valid_selected)[:, 1],
        validation_catboost.predict_proba(X_valid_selected)[:, 1],
        BLEND_LIGHTGBM_WEIGHT,
    )
    threshold_info = find_best_threshold(y.loc[valid_index], validation_probabilities)
    selected_threshold = float(threshold_info["threshold"])

    print("Second-order blend: training final holdout models")
    X_train_final, X_test, target_encoding_mappings = build_model_frames(
        reduced=reduced,
        recent_domain=augmented_recent,
        train_df=train_df,
        previous=previous_te,
        y=y,
        train_index=train_index,
        transform_index=test_index,
        oof_train=True,
    )
    (
        X_train_final_selected,
        X_test_selected,
        final_preprocessor,
        final_mask,
        final_names,
    ) = selected_transformed_matrices(
        X_train_final,
        y.loc[train_index],
        X_test,
    )
    final_lightgbm = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
    final_lightgbm.fit(X_train_final_selected, y.loc[train_index])
    final_catboost = fit_catboost(
        X_train_final_selected,
        y.loc[train_index].to_numpy(),
    )
    test_probabilities = make_blend_probabilities(
        final_lightgbm.predict_proba(X_test_selected)[:, 1],
        final_catboost.predict_proba(X_test_selected)[:, 1],
        BLEND_LIGHTGBM_WEIGHT,
    )
    holdout_metrics = pd.DataFrame(
        [
            {
                "model": "final_second_order_advanced_blend",
                "threshold_strategy": "default_0.5",
                "lightgbm_weight": BLEND_LIGHTGBM_WEIGHT,
                "catboost_weight": 1.0 - BLEND_LIGHTGBM_WEIGHT,
                **evaluate_probabilities(y.loc[test_index], test_probabilities, 0.5),
            },
            {
                "model": "final_second_order_advanced_blend",
                "threshold_strategy": "validation_selected",
                "lightgbm_weight": BLEND_LIGHTGBM_WEIGHT,
                "catboost_weight": 1.0 - BLEND_LIGHTGBM_WEIGHT,
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
            "lightgbm_model": final_lightgbm,
            "catboost_model": final_catboost,
            "preprocessor": final_preprocessor,
            "selected_mask": final_mask,
            "transformed_feature_names": final_names,
            "selected_transformed_features": final_names[final_mask],
            "dropped_transformed_features": sorted(DROPPED_TRANSFORMED_FEATURES),
            "raw_feature_columns": X_train_final.columns.tolist(),
            "target_encoding_mappings": target_encoding_mappings,
            "threshold": selected_threshold,
            "lightgbm_weight": BLEND_LIGHTGBM_WEIGHT,
            "catboost_weight": 1.0 - BLEND_LIGHTGBM_WEIGHT,
            "lightgbm_params": OPTUNA_LIGHTGBM_PARAMS,
            "catboost_params": CATBOOST_PARAMS,
            "second_order_feature_columns": build_second_order_aligned_features(
                train_df
            ).columns.tolist(),
        },
        MODEL_PATH,
        compress=3,
    )

    return holdout_metrics, confusion, classification, selected_threshold


def write_report(
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    second_order_columns: list[str],
    holdout_metrics: pd.DataFrame | None,
    confusion: np.ndarray | None,
    classification: str | None,
    selected_threshold: float | None,
) -> None:
    second_order_row = cv_summary[
        cv_summary["feature_set"] == "advanced_drop_2_plus_second_order"
    ].iloc[0]
    cv_delta = pd.DataFrame(
        [
            {
                "metric": "mean_cv_roc_auc",
                "delta": second_order_row["mean_cv_roc_auc"] - BASELINE_CV_ROC_AUC,
            },
            {
                "metric": "mean_cv_average_precision",
                "delta": second_order_row["mean_cv_average_precision"]
                - BASELINE_CV_AVERAGE_PRECISION,
            },
        ]
    )
    lines = [
        "# Second-Order Relational Features Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Setup",
        "",
        "- Added compact second-order recent-history features with `SO_` prefixes in `src/relational_features.py`.",
        "- Appended those features to the current advanced recent/domain block on top of the top-200 reduced base features.",
        "- Kept the same fold-safe target encoding and transformed-feature pruning approach.",
        "- LightGBM CV uses the same 3-fold stratified protocol as `run_advanced_relational_cv.py`.",
        f"- Baseline: `advanced_drop_2_weakest_transformed`, CV ROC-AUC `{BASELINE_CV_ROC_AUC:.6f}`, AP `{BASELINE_CV_AVERAGE_PRECISION:.6f}`.",
        "",
        "## Added Second-Order Features",
        "",
        as_markdown(pd.DataFrame({"feature": second_order_columns})),
        "",
        "## LightGBM CV Results",
        "",
        as_markdown(cv_results),
        "",
        "## CV Summary",
        "",
        as_markdown(cv_summary),
        "",
        "Delta versus baseline:",
        "",
        as_markdown(cv_delta),
        "",
    ]

    if holdout_metrics is None:
        lines.extend(
            [
                "The second-order LightGBM CV ROC-AUC did not improve over the baseline, so the 60/40 LightGBM+CatBoost holdout blend was not retrained.",
                "",
            ]
        )
    else:
        selected_metrics = holdout_metrics[
            holdout_metrics["threshold_strategy"] == "validation_selected"
        ].iloc[0]
        comparison = pd.DataFrame(
            [
                BASELINE_HOLDOUT_BLEND,
                {
                    "model": "final_second_order_advanced_blend",
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
                "The second-order LightGBM CV ROC-AUC improved, so the 60/40 LightGBM+CatBoost blend was retrained on the new feature set.",
                "",
                f"Selected holdout threshold: `{selected_threshold:.6f}`",
                "",
                "## Holdout Metrics",
                "",
                as_markdown(holdout_metrics),
                "",
                "## Holdout Comparison Against Current Blend",
                "",
                as_markdown(comparison),
                "",
                "Delta, second-order blend minus current blend:",
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

    print("Building reduced, advanced, and second-order feature inputs")
    reduced = build_reduced_matrix(train_df)
    augmented_recent = build_augmented_recent_features(train_df)
    second_order = build_second_order_aligned_features(train_df)
    previous_te = load_previous_te_rows()

    print("Running 3-fold LightGBM CV")
    cv_results = run_lightgbm_cv(
        reduced=reduced,
        augmented_recent=augmented_recent,
        train_df=train_df,
        previous_te=previous_te,
        y=y,
    )
    cv_summary = summarize_cv(cv_results)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    cv_summary.to_csv(
        REPORTS_DIR / "second_order_relational_features_cv_summary.csv",
        index=False,
    )

    holdout_metrics = None
    confusion = None
    classification = None
    selected_threshold = None
    second_order_auc = float(
        cv_summary.loc[
            cv_summary["feature_set"] == "advanced_drop_2_plus_second_order",
            "mean_cv_roc_auc",
        ].iloc[0]
    )
    if second_order_auc > BASELINE_CV_ROC_AUC:
        print("CV improved; retraining 60/40 LightGBM+CatBoost blend")
        holdout_metrics, confusion, classification, selected_threshold = (
            train_second_order_blend(
                train_df=train_df,
                y=y,
                reduced=reduced,
                augmented_recent=augmented_recent,
                previous_te=previous_te,
            )
        )
        holdout_metrics.to_csv(HOLDOUT_METRICS_PATH, index=False)
        pd.DataFrame(
            confusion,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(CONFUSION_PATH)
        CLASSIFICATION_PATH.write_text(classification, encoding="utf-8")
    else:
        print("CV did not improve; skipping final blend retrain")

    write_report(
        cv_results=cv_results,
        cv_summary=cv_summary,
        second_order_columns=second_order.columns.tolist(),
        holdout_metrics=holdout_metrics,
        confusion=confusion,
        classification=classification,
        selected_threshold=selected_threshold,
    )

    print(cv_results.to_string(index=False))
    print(cv_summary.to_string(index=False))
    if holdout_metrics is not None:
        print(holdout_metrics.to_string(index=False))
        print(confusion)
        print(f"Saved {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
