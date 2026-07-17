from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
    OPTUNA_LIGHTGBM_PARAMS,
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
MODEL_PATH = MODELS_DIR / "final_advanced_blend.joblib"
REPORT_PATH = REPORTS_DIR / "final_advanced_blend_report.md"
CATBOOST_CV_PATH = REPORTS_DIR / "advanced_blend_catboost_cv_results.csv"
BLEND_CV_PATH = REPORTS_DIR / "advanced_blend_cv_results.csv"
HOLDOUT_METRICS_PATH = REPORTS_DIR / "final_advanced_blend_holdout_metrics.csv"
CONFUSION_PATH = REPORTS_DIR / "final_advanced_blend_confusion_matrix.csv"
CLASSIFICATION_PATH = REPORTS_DIR / "final_advanced_blend_classification_report.txt"

BASELINE_CV_ROC_AUC = 0.7925488471724659
BASELINE_HOLDOUT = {
    "model": "final_advanced_pruned_lgbm",
    "threshold": 0.670437625096521,
    "roc_auc": 0.794353430966071,
    "average_precision": 0.2967048111538493,
    "accuracy": 0.867112173389916,
    "precision_class_1": 0.28939075630252103,
    "recall_class_1": 0.44390735146022153,
    "f1_class_1": 0.3503696049598601,
}
BLEND_WEIGHTS = [0.50, 0.60, 0.70, 0.80]


CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 1200,
    "learning_rate": 0.035,
    "depth": 5,
    "l2_leaf_reg": 8.0,
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": -1,
    "boosting_type": "Plain",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "one_hot_max_size": 20,
    "max_ctr_complexity": 1,
    "od_type": "Iter",
    "od_wait": 100,
}


def blend_name(lightgbm_weight: float) -> str:
    lightgbm_pct = int(round(lightgbm_weight * 100))
    catboost_pct = 100 - lightgbm_pct

    return f"lgbm_{lightgbm_pct}_catboost_{catboost_pct}"


def selected_transformed_matrices(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[object, object, object, np.ndarray, np.ndarray]:
    preprocessor = make_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_valid_transformed = preprocessor.transform(X_valid)
    transformed_names = np.asarray(preprocessor.get_feature_names_out())
    selected_mask = np.asarray(
        [name not in DROPPED_TRANSFORMED_FEATURES for name in transformed_names]
    )

    return (
        X_train_transformed[:, selected_mask],
        X_valid_transformed[:, selected_mask],
        preprocessor,
        selected_mask,
        transformed_names,
    )


def fit_catboost(
    X_train,
    y_train: np.ndarray,
    X_valid=None,
    y_valid: np.ndarray | None = None,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    fit_kwargs = {}
    if y_valid is not None:
        fit_kwargs["eval_set"] = Pool(X_valid, y_valid)
        fit_kwargs["use_best_model"] = True
    model.fit(Pool(X_train, y_train), **fit_kwargs)

    return model


def make_blend_probabilities(
    lightgbm_probabilities: np.ndarray,
    catboost_probabilities: np.ndarray,
    lightgbm_weight: float,
) -> np.ndarray:
    return (
        lightgbm_weight * lightgbm_probabilities
        + (1.0 - lightgbm_weight) * catboost_probabilities
    )


def run_cv(
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    train_df: pd.DataFrame,
    previous_te: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    catboost_rows = []
    lightgbm_rows = []
    blend_rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
        train_index = reduced.index.to_numpy()[train_pos]
        valid_index = reduced.index.to_numpy()[valid_pos]
        print(f"Fold {fold}: building advanced_drop_2 transformed matrices")
        X_train, X_valid = build_advanced_fold_frames(
            reduced=reduced,
            recent_domain=recent_domain,
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
        y_train = y.loc[train_index].to_numpy()
        y_valid = y.loc[valid_index].to_numpy()

        print(f"Fold {fold}: training LightGBM side")
        lightgbm = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
        lightgbm.fit(X_train_selected, y_train)
        lightgbm_probabilities = lightgbm.predict_proba(X_valid_selected)[:, 1]
        lightgbm_rows.append(
            {
                "model": "lightgbm_advanced_pruned",
                "fold": fold,
                "transformed_feature_count": int(selected_mask.sum()),
                "roc_auc": roc_auc_score(y_valid, lightgbm_probabilities),
                "average_precision": average_precision_score(
                    y_valid,
                    lightgbm_probabilities,
                ),
            }
        )

        print(f"Fold {fold}: training CatBoost side")
        catboost = fit_catboost(
            X_train_selected,
            y_train,
            X_valid_selected,
            y_valid,
        )
        catboost_probabilities = catboost.predict_proba(X_valid_selected)[:, 1]
        catboost_rows.append(
            {
                "model": "catboost_advanced_drop_2",
                "fold": fold,
                "best_iteration": int(catboost.best_iteration_ or CATBOOST_PARAMS["iterations"]),
                "transformed_feature_count": int(selected_mask.sum()),
                "roc_auc": roc_auc_score(y_valid, catboost_probabilities),
                "average_precision": average_precision_score(
                    y_valid,
                    catboost_probabilities,
                ),
            }
        )

        for lightgbm_weight in BLEND_WEIGHTS:
            probabilities = make_blend_probabilities(
                lightgbm_probabilities,
                catboost_probabilities,
                lightgbm_weight,
            )
            blend_rows.append(
                {
                    "blend": blend_name(lightgbm_weight),
                    "lightgbm_weight": lightgbm_weight,
                    "catboost_weight": 1.0 - lightgbm_weight,
                    "fold": fold,
                    "roc_auc": roc_auc_score(y_valid, probabilities),
                    "average_precision": average_precision_score(
                        y_valid,
                        probabilities,
                    ),
                }
            )

        del (
            X_train,
            X_valid,
            X_train_selected,
            X_valid_selected,
            lightgbm,
            catboost,
        )
        gc.collect()

    return (
        pd.DataFrame(catboost_rows),
        pd.DataFrame(lightgbm_rows),
        pd.DataFrame(blend_rows),
    )


def summarize_blends(blend_cv: pd.DataFrame) -> pd.DataFrame:
    blend_cv = blend_cv.copy()
    blend_cv["blend"] = blend_cv["lightgbm_weight"].map(blend_name)

    return (
        blend_cv.groupby(["blend", "lightgbm_weight", "catboost_weight"], as_index=False)
        .agg(
            mean_cv_roc_auc=("roc_auc", "mean"),
            std_cv_roc_auc=("roc_auc", lambda values: values.std(ddof=0)),
            mean_cv_average_precision=("average_precision", "mean"),
            std_cv_average_precision=(
                "average_precision",
                lambda values: values.std(ddof=0),
            ),
        )
        .sort_values(["mean_cv_roc_auc", "mean_cv_average_precision"], ascending=False)
    )


def train_final_blend(
    train_df: pd.DataFrame,
    y: pd.Series,
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    previous_te: pd.DataFrame,
    lightgbm_weight: float,
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

    print("Final blend: selecting threshold on validation split")
    X_train_full, X_valid, _ = build_model_frames(
        reduced=reduced,
        recent_domain=recent_domain,
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
        lightgbm_weight,
    )
    threshold_info = find_best_threshold(y.loc[valid_index], validation_probabilities)
    selected_threshold = float(threshold_info["threshold"])

    print("Final blend: training final holdout models")
    X_train_final, X_test, target_encoding_mappings = build_model_frames(
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
        lightgbm_weight,
    )
    holdout_metrics = pd.DataFrame(
        [
            {
                "model": "final_advanced_blend",
                "threshold_strategy": "default_0.5",
                "lightgbm_weight": lightgbm_weight,
                "catboost_weight": 1.0 - lightgbm_weight,
                **evaluate_probabilities(y.loc[test_index], test_probabilities, 0.5),
            },
            {
                "model": "final_advanced_blend",
                "threshold_strategy": "validation_selected",
                "lightgbm_weight": lightgbm_weight,
                "catboost_weight": 1.0 - lightgbm_weight,
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
            "lightgbm_weight": lightgbm_weight,
            "catboost_weight": 1.0 - lightgbm_weight,
            "lightgbm_params": OPTUNA_LIGHTGBM_PARAMS,
            "catboost_params": CATBOOST_PARAMS,
        },
        MODEL_PATH,
        compress=3,
    )

    return holdout_metrics, confusion, classification, selected_threshold


def write_report(
    catboost_cv: pd.DataFrame,
    lightgbm_cv: pd.DataFrame,
    blend_cv: pd.DataFrame,
    blend_summary: pd.DataFrame,
    best_blend_row: pd.Series,
    holdout_metrics: pd.DataFrame | None,
    confusion: np.ndarray | None,
    classification: str | None,
    selected_threshold: float | None,
) -> None:
    catboost_summary = pd.DataFrame(
        [
            {
                "mean_cv_roc_auc": catboost_cv["roc_auc"].mean(),
                "std_cv_roc_auc": catboost_cv["roc_auc"].std(ddof=0),
                "mean_cv_average_precision": catboost_cv["average_precision"].mean(),
                "std_cv_average_precision": catboost_cv["average_precision"].std(ddof=0),
            }
        ]
    )
    lightgbm_summary = pd.DataFrame(
        [
            {
                "mean_cv_roc_auc": lightgbm_cv["roc_auc"].mean(),
                "std_cv_roc_auc": lightgbm_cv["roc_auc"].std(ddof=0),
                "mean_cv_average_precision": lightgbm_cv["average_precision"].mean(),
                "std_cv_average_precision": lightgbm_cv["average_precision"].std(ddof=0),
            }
        ]
    )
    lines = [
        "# Final Advanced Blend Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Setup",
        "",
        "- CatBoost was trained on the same `advanced_drop_2_weakest_transformed` matrix as the advanced pruned LightGBM.",
        "- LightGBM side uses the untuned advanced-pruned parameters from `final_advanced_pruned_lgbm`, not the later Optuna-tuned model.",
        "- CV uses the same `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` protocol as `run_advanced_relational_cv.py`.",
        "- Target encodings are fold-safe and generated with the same helper functions as the existing advanced CV scripts.",
        f"- Blend ratios tested: `{', '.join(blend_name(w).replace('lgbm_', '').replace('_catboost_', '/') for w in BLEND_WEIGHTS)}` LightGBM/CatBoost.",
        f"- Baseline CV ROC-AUC to beat: `{BASELINE_CV_ROC_AUC:.6f}`",
        "",
        "## CatBoost CV Metrics",
        "",
        as_markdown(catboost_cv),
        "",
        "CatBoost summary:",
        "",
        as_markdown(catboost_summary),
        "",
        "## LightGBM CV Metrics",
        "",
        as_markdown(lightgbm_cv),
        "",
        "LightGBM summary:",
        "",
        as_markdown(lightgbm_summary),
        "",
        "## Blend CV Metrics",
        "",
        as_markdown(blend_cv),
        "",
        "## Blend CV Summary",
        "",
        as_markdown(blend_summary),
        "",
        f"Best blend: `{best_blend_row['blend']}` with mean CV ROC-AUC `{best_blend_row['mean_cv_roc_auc']:.6f}`.",
        "",
    ]

    if holdout_metrics is None:
        lines.extend(
            [
                "No blend beat the baseline CV ROC-AUC, so no final blend model was trained or saved.",
                "",
            ]
        )
    else:
        selected_metrics = holdout_metrics[
            holdout_metrics["threshold_strategy"] == "validation_selected"
        ].iloc[0]
        comparison = pd.DataFrame(
            [
                BASELINE_HOLDOUT,
                {
                    "model": "final_advanced_blend",
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
        lines.extend(
            [
                f"Selected holdout threshold: `{selected_threshold:.6f}`",
                "",
                "## Holdout Metrics",
                "",
                as_markdown(holdout_metrics),
                "",
                "## Holdout Comparison",
                "",
                as_markdown(comparison),
                "",
                "Delta, blend minus previous advanced pruned LightGBM:",
                "",
                as_markdown(
                    delta.rename("delta").reset_index().rename(columns={"index": "metric"})
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

    print("Building advanced feature inputs")
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()

    lightgbm_cv_path = REPORTS_DIR / "advanced_blend_lightgbm_cv_results.csv"
    if CATBOOST_CV_PATH.exists() and BLEND_CV_PATH.exists() and lightgbm_cv_path.exists():
        print("Loading existing completed blend CV results")
        catboost_cv = pd.read_csv(CATBOOST_CV_PATH)
        lightgbm_cv = pd.read_csv(lightgbm_cv_path)
        blend_cv = pd.read_csv(BLEND_CV_PATH)
        blend_cv["blend"] = blend_cv["lightgbm_weight"].map(blend_name)
    else:
        catboost_cv, lightgbm_cv, blend_cv = run_cv(
            reduced=reduced,
            recent_domain=recent_domain,
            train_df=train_df,
            previous_te=previous_te,
            y=y,
        )
    catboost_cv.to_csv(CATBOOST_CV_PATH, index=False)
    blend_cv.to_csv(BLEND_CV_PATH, index=False)
    lightgbm_cv.to_csv(lightgbm_cv_path, index=False)

    blend_summary = summarize_blends(blend_cv)
    blend_summary.to_csv(
        REPORTS_DIR / "advanced_blend_cv_summary.csv",
        index=False,
    )
    best_blend_row = blend_summary.iloc[0]

    holdout_metrics = None
    confusion = None
    classification = None
    selected_threshold = None
    if float(best_blend_row["mean_cv_roc_auc"]) > BASELINE_CV_ROC_AUC:
        print("Best blend beat baseline CV ROC-AUC; training final holdout blend")
        holdout_metrics, confusion, classification, selected_threshold = train_final_blend(
            train_df=train_df,
            y=y,
            reduced=reduced,
            recent_domain=recent_domain,
            previous_te=previous_te,
            lightgbm_weight=float(best_blend_row["lightgbm_weight"]),
        )
        holdout_metrics.to_csv(HOLDOUT_METRICS_PATH, index=False)
        pd.DataFrame(
            confusion,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(CONFUSION_PATH)
        CLASSIFICATION_PATH.write_text(classification, encoding="utf-8")
    else:
        print("No blend beat baseline CV ROC-AUC; skipping final blend training")

    write_report(
        catboost_cv=catboost_cv,
        lightgbm_cv=lightgbm_cv,
        blend_cv=blend_cv,
        blend_summary=blend_summary,
        best_blend_row=best_blend_row,
        holdout_metrics=holdout_metrics,
        confusion=confusion,
        classification=classification,
        selected_threshold=selected_threshold,
    )

    print(catboost_cv.to_string(index=False))
    print(blend_summary.to_string(index=False))
    if holdout_metrics is not None:
        print(holdout_metrics.to_string(index=False))
        print(confusion)
        print(f"Saved {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
