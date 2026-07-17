from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_blend import (
    make_blend_probabilities,
    selected_transformed_matrices,
)
from scripts.run_advanced_feature_pruning import build_advanced_fold_frames
from scripts.run_advanced_relational_cv import (
    OPTUNA_LIGHTGBM_PARAMS,
    build_recent_domain_features,
    build_reduced_matrix,
    load_previous_te_rows,
)
from scripts.run_native_catboost_categorical import (
    build_native_catboost_matrix,
    fit_native_catboost,
)
from scripts.run_relational_feature_experiments import (
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from scripts.run_relational_feature_pruning import as_markdown


ROBUSTNESS_SEEDS = [42, 123, 2024]
ROBUSTNESS_SPLITS = 5
BLEND_LIGHTGBM_WEIGHT = 0.50

ROBUSTNESS_RESULTS_PATH = REPORTS_DIR / "final_native_blend_robustness_5fold_results.csv"
ROBUSTNESS_SUMMARY_PATH = REPORTS_DIR / "final_native_blend_robustness_5fold_summary.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_model_summary_report.md"


def run_robustness_cv(
    train_df: pd.DataFrame,
    y: pd.Series,
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    previous_te: pd.DataFrame,
    native_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for seed in ROBUSTNESS_SEEDS:
        cv = StratifiedKFold(
            n_splits=ROBUSTNESS_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
            train_index = reduced.index.to_numpy()[train_pos]
            valid_index = reduced.index.to_numpy()[valid_pos]
            print(f"seed={seed} fold={fold}: LightGBM")
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

            print(f"seed={seed} fold={fold}: native CatBoost")
            catboost_probabilities, best_iteration, categorical_columns = fit_native_catboost(
                native_matrix.loc[train_index],
                y.loc[train_index],
                native_matrix.loc[valid_index],
                y.loc[valid_index],
            )
            blend_probabilities = make_blend_probabilities(
                lgbm_probabilities,
                catboost_probabilities,
                BLEND_LIGHTGBM_WEIGHT,
            )
            y_valid = y.loc[valid_index]
            rows.extend(
                [
                    {
                        "model": "lightgbm_component",
                        "seed": seed,
                        "fold": fold,
                        "transformed_feature_count": int(selected_mask.sum()),
                        "catboost_best_iteration": np.nan,
                        "catboost_categorical_feature_count": np.nan,
                        "roc_auc": roc_auc_score(y_valid, lgbm_probabilities),
                        "average_precision": average_precision_score(
                            y_valid,
                            lgbm_probabilities,
                        ),
                    },
                    {
                        "model": "native_catboost_component",
                        "seed": seed,
                        "fold": fold,
                        "transformed_feature_count": np.nan,
                        "catboost_best_iteration": best_iteration,
                        "catboost_categorical_feature_count": len(categorical_columns),
                        "roc_auc": roc_auc_score(y_valid, catboost_probabilities),
                        "average_precision": average_precision_score(
                            y_valid,
                            catboost_probabilities,
                        ),
                    },
                    {
                        "model": "native_50_50_blend",
                        "seed": seed,
                        "fold": fold,
                        "transformed_feature_count": int(selected_mask.sum()),
                        "catboost_best_iteration": best_iteration,
                        "catboost_categorical_feature_count": len(categorical_columns),
                        "roc_auc": roc_auc_score(y_valid, blend_probabilities),
                        "average_precision": average_precision_score(
                            y_valid,
                            blend_probabilities,
                        ),
                    },
                ]
            )
            pd.DataFrame(rows).to_csv(ROBUSTNESS_RESULTS_PATH, index=False)
            del (
                X_lgbm_train,
                X_lgbm_valid,
                X_lgbm_train_selected,
                X_lgbm_valid_selected,
                lightgbm,
            )
            gc.collect()

    return pd.DataFrame(rows)


def summarize_robustness(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("model", as_index=False)
        .agg(
            folds=("roc_auc", "count"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            min_roc_auc=("roc_auc", "min"),
            max_roc_auc=("roc_auc", "max"),
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            min_average_precision=("average_precision", "min"),
            max_average_precision=("average_precision", "max"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )


def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def write_final_summary_report(
    robustness_results: pd.DataFrame,
    robustness_summary: pd.DataFrame,
) -> None:
    robustness_blend = robustness_summary[
        robustness_summary["model"] == "native_50_50_blend"
    ].iloc[0]
    robustness_lgbm = robustness_summary[
        robustness_summary["model"] == "lightgbm_component"
    ].iloc[0]
    robustness_delta = pd.DataFrame(
        [
            {
                "metric": "mean_roc_auc",
                "blend_minus_lightgbm": robustness_blend["mean_roc_auc"]
                - robustness_lgbm["mean_roc_auc"],
            },
            {
                "metric": "mean_average_precision",
                "blend_minus_lightgbm": robustness_blend[
                    "mean_average_precision"
                ]
                - robustness_lgbm["mean_average_precision"],
            },
        ]
    )
    native_holdout = load_optional_csv(
        REPORTS_DIR / "final_native_catboost_blend_holdout_metrics.csv"
    )
    stacked_holdout = load_optional_csv(
        REPORTS_DIR / "final_stacked_ensemble_holdout_metrics.csv"
    )
    native_selected = native_holdout[
        native_holdout["threshold_strategy"] == "validation_selected"
    ]
    stacked_selected = stacked_holdout[
        stacked_holdout["threshold_strategy"] == "validation_selected"
    ]
    holdout_comparison = pd.concat(
        [
            native_selected.assign(model_label="final_native_catboost_blend"),
            stacked_selected.assign(model_label="final_stacked_ensemble"),
        ],
        ignore_index=True,
    )
    if not holdout_comparison.empty:
        holdout_comparison = holdout_comparison[
            [
                "model_label",
                "threshold",
                "roc_auc",
                "average_precision",
                "accuracy",
                "precision_class_1",
                "recall_class_1",
                "f1_class_1",
            ]
        ]

    lines = [
        "# Final Model Summary Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Objective",
        "",
        "The project predicts Home Credit default risk using application data and relational history tables. The primary ranking metric is ROC-AUC, with average precision and threshold-based class-1 F1 used to check behavior on the imbalanced positive class.",
        "",
        "## Data And Feature Sources",
        "",
        "- Main table: `application_train.csv` keyed by `SK_ID_CURR`.",
        "- Relational tables: bureau, bureau balance, previous applications, installments, POS cash balance, and credit-card balance.",
        "- Joining strategy: aggregate relational histories to one row per applicant, then left-join to the application row.",
        "- Final native CatBoost side keeps raw categorical application fields plus previous-application categorical summaries; LightGBM side uses the advanced pruned transformed feature pipeline.",
        "",
        "## Model Selection Journey",
        "",
        "| Stage | Main decision | Result / consequence |",
        "| --- | --- | --- |",
        "| Application-only baseline | Started from engineered application features only | Holdout ROC-AUC `0.769069`; useful but below target. |",
        "| Full relational joins | Added bureau, previous, installments, POS, and credit-card aggregates | Holdout ROC-AUC rose to about `0.790203`; relational history was the largest lift. |",
        "| Feature reduction | Reduced 908 raw / 1884 transformed relational features to top-200 base features | Similar ROC-AUC with far fewer transformed features; reduced over-wide one-hot/aggregate noise. |",
        "| LightGBM tuning | Tuned reduced LightGBM with Optuna | Holdout ROC-AUC `0.791323`; modest but real improvement. |",
        "| Recent/domain features | Added recent-window and domain-specific relational features with fold-safe target encoding | 3-fold CV ROC-AUC improved to `0.792274`; recent behavior helped. |",
        "| Transformed-feature pruning | Dropped weakest transformed occupation one-hot features | Selected `advanced_drop_2_weakest_transformed`; 3-fold CV ROC-AUC `0.792549`. |",
        "| Final advanced pruned LightGBM | Trained holdout model with advanced-drop-2 features | Holdout ROC-AUC `0.794353`, AP `0.296705`, F1 `0.350370`. |",
        "| Advanced LightGBM Optuna | Tuned the advanced-pruned feature set | Holdout ROC-AUC `0.794835`, but AP dipped slightly; not a decisive replacement. |",
        "| CatBoost on transformed features | Tested CatBoost on LightGBM-style transformed matrix | CV ROC-AUC `0.791913`; below LightGBM/blend. |",
        "| Weighted blend | Blended advanced LightGBM with transformed CatBoost | 60/40 blend holdout ROC-AUC `0.795338`, AP `0.298163`, F1 `0.352544`. |",
        "| Second-order recent features | Added compact ratio/trend recent-history features | CV ROC-AUC did not improve (`0.792518` vs `0.792549`), so not promoted. |",
        "| Native categorical CatBoost | Replaced transformed CatBoost side with native categorical CatBoost | CatBoost CV ROC-AUC improved to `0.793058`; 50/50 blend CV ROC-AUC improved to `0.794531`. |",
        "| Final native blend | Trained 50/50 LightGBM/native-CatBoost blend | Holdout ROC-AUC `0.796126`, AP `0.299074`, F1 `0.350203`; best AP and strong ROC-AUC. |",
        "| Stacking | Logistic regression on base OOF predictions plus top raw features | CV ROC-AUC `0.794617` and holdout ROC-AUC `0.796508`, but AP and F1 dropped; not selected as champion. |",
        "",
        "## Robustness Check",
        "",
        "The final robustness check re-ran 5-fold stratified CV across seeds `42`, `123`, and `2024`, producing 15 fold/seed evaluations per model component.",
        "",
        as_markdown(robustness_summary),
        "",
        "Blend advantage over LightGBM component:",
        "",
        as_markdown(robustness_delta),
        "",
        "Per-fold robustness metrics:",
        "",
        as_markdown(robustness_results),
        "",
        "## Final Holdout Comparison",
        "",
        as_markdown(holdout_comparison),
        "",
        "## Recommended Final Model",
        "",
        "Recommended model: `final_native_catboost_blend`, a 50/50 weighted average of:",
        "",
        "- advanced-pruned LightGBM using top-200 reduced base features, recent/domain relational features, fold-safe target encodings, and transformed-feature pruning",
        "- native-categorical CatBoost using the same numeric/relational features plus raw categorical columns passed through `cat_features`",
        "",
        "Saved artifact: `models/final_native_catboost_blend.joblib`",
        "",
        "Key final holdout metrics:",
        "",
        "- ROC-AUC: `0.796126`",
        "- Average precision: `0.299074`",
        "- Accuracy: `0.864722`",
        "- Class-1 precision: `0.286006`",
        "- Class-1 recall: `0.451561`",
        "- Class-1 F1: `0.350203`",
        "- Selected threshold: `0.668978`",
        "",
        "## Why This Model Over Stacking",
        "",
        "The stacked ensemble had the highest holdout ROC-AUC (`0.796508`), but the gain over the native blend was small (`+0.000382`) and came with worse AP (`0.296714` vs `0.299074`) and worse class-1 F1 (`0.348905` vs `0.350203`). Since the problem is imbalanced, AP and class-1 F1 are important practical checks. The native 50/50 blend is also simpler and more stable operationally than the logistic stack, while preserving nearly the same ROC-AUC and better positive-class quality.",
        "",
        "## Final Artifacts",
        "",
        "- `models/final_native_catboost_blend.joblib`",
        "- `reports/final_native_catboost_blend_report.md`",
        "- `reports/final_native_blend_robustness_5fold_results.csv`",
        "- `reports/final_native_blend_robustness_5fold_summary.csv`",
        "- `reports/final_model_summary_report.md`",
        "- Supporting experiment scripts in `scripts/` for reproducibility",
        "",
    ]
    FINAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Reading application data")
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    print("Building final model feature matrices")
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()
    native_matrix = build_native_catboost_matrix(train_df)

    print("Running robustness CV")
    results = run_robustness_cv(
        train_df=train_df,
        y=y,
        reduced=reduced,
        recent_domain=recent_domain,
        previous_te=previous_te,
        native_matrix=native_matrix,
    )
    summary = summarize_robustness(results)
    results.to_csv(ROBUSTNESS_RESULTS_PATH, index=False)
    summary.to_csv(ROBUSTNESS_SUMMARY_PATH, index=False)

    write_final_summary_report(results, summary)

    print(summary.to_string(index=False))
    print(f"Wrote {ROBUSTNESS_RESULTS_PATH}")
    print(f"Wrote {ROBUSTNESS_SUMMARY_PATH}")
    print(f"Wrote {FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
