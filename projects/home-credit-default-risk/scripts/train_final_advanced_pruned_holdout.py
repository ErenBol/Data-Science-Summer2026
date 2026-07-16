from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_feature_pruning import make_preprocessor
from scripts.run_advanced_relational_cv import (
    APPLICATION_TE_COLUMNS,
    INNER_TE_SPLITS,
    OPTUNA_LIGHTGBM_PARAMS,
    PREVIOUS_TE_COLUMNS,
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
from src.thresholding import evaluate_probabilities, find_best_threshold


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_advanced_pruned_lgbm.joblib"
REPORT_PATH = REPORTS_DIR / "final_advanced_pruned_holdout_report.md"

DROPPED_TRANSFORMED_FEATURES = {
    "categorical__OCCUPATION_TYPE_infrequent_sklearn",
    "categorical__OCCUPATION_TYPE_Security staff",
}


def _fit_category_mapping(
    values: pd.Series,
    target: pd.Series,
    min_count: int = 50,
) -> tuple[pd.Series, float]:
    global_mean = float(target.mean())
    stats = (
        pd.DataFrame({"category": values.astype("object"), "target": target})
        .dropna(subset=["category"])
        .groupby("category")["target"]
        .agg(["mean", "count"])
    )
    mapping = stats.loc[stats["count"] >= min_count, "mean"]

    return mapping, global_mean


def _transform_application_te(
    train_df: pd.DataFrame,
    fit_index: np.ndarray,
    transform_index: np.ndarray,
    mapping: pd.Series,
    global_mean: float,
    column: str,
) -> pd.Series:
    return (
        train_df.loc[transform_index, column]
        .map(mapping)
        .fillna(global_mean)
        .rename(f"TE_APP_{column}")
    )


def _fit_previous_mapping(
    previous: pd.DataFrame,
    train_df: pd.DataFrame,
    y: pd.Series,
    fit_index: np.ndarray,
    column: str,
) -> tuple[pd.Series, float]:
    fit_targets = train_df.loc[fit_index, ["SK_ID_CURR"]].copy()
    fit_targets["TARGET"] = y.loc[fit_index].to_numpy()
    fit_previous = previous[["SK_ID_CURR", column]].merge(
        fit_targets,
        on="SK_ID_CURR",
        how="inner",
    )

    return _fit_category_mapping(fit_previous[column], fit_previous["TARGET"])


def _transform_previous_te(
    previous: pd.DataFrame,
    train_df: pd.DataFrame,
    transform_index: np.ndarray,
    mapping: pd.Series,
    global_mean: float,
    column: str,
) -> pd.DataFrame:
    transform_ids = train_df.loc[transform_index, ["SK_ID_CURR"]]
    transform_previous = previous[["SK_ID_CURR", column]].merge(
        transform_ids,
        on="SK_ID_CURR",
        how="inner",
    )
    transform_previous["ENCODED"] = (
        transform_previous[column].map(mapping).fillna(global_mean)
    )

    grouped = transform_previous.groupby("SK_ID_CURR")["ENCODED"]
    features = pd.DataFrame(index=transform_ids["SK_ID_CURR"])
    prefix = f"TE_PREV_{column}"
    features[f"{prefix}_MEAN"] = grouped.mean()
    features[f"{prefix}_MAX"] = grouped.max()
    features[f"{prefix}_COUNT"] = grouped.size()
    features = features.fillna(
        {
            f"{prefix}_MEAN": global_mean,
            f"{prefix}_MAX": global_mean,
            f"{prefix}_COUNT": 0,
        }
    )
    features.index = transform_index

    return features


def build_target_encoding_frames(
    train_df: pd.DataFrame,
    previous: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    transform_index: np.ndarray,
    oof_train: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_features = pd.DataFrame(index=train_index)
    transform_features = pd.DataFrame(index=transform_index)
    mappings: dict = {"application": {}, "previous": {}}

    inner_cv = StratifiedKFold(
        n_splits=INNER_TE_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for column in APPLICATION_TE_COLUMNS:
        output_column = f"TE_APP_{column}"
        train_features[output_column] = np.nan

        if oof_train:
            for fit_pos, holdout_pos in inner_cv.split(train_index, y.loc[train_index]):
                fit_index = train_index[fit_pos]
                holdout_index = train_index[holdout_pos]
                mapping, global_mean = _fit_category_mapping(
                    train_df.loc[fit_index, column],
                    y.loc[fit_index],
                )
                train_features.loc[holdout_index, output_column] = (
                    _transform_application_te(
                        train_df,
                        fit_index,
                        holdout_index,
                        mapping,
                        global_mean,
                        column,
                    )
                )
        else:
            mapping, global_mean = _fit_category_mapping(
                train_df.loc[train_index, column],
                y.loc[train_index],
            )
            train_features[output_column] = _transform_application_te(
                train_df,
                train_index,
                train_index,
                mapping,
                global_mean,
                column,
            )

        mapping, global_mean = _fit_category_mapping(
            train_df.loc[train_index, column],
            y.loc[train_index],
        )
        mappings["application"][column] = {
            "mapping": mapping.to_dict(),
            "global_mean": global_mean,
        }
        transform_features[output_column] = _transform_application_te(
            train_df,
            train_index,
            transform_index,
            mapping,
            global_mean,
            column,
        )

    for column in PREVIOUS_TE_COLUMNS:
        output_columns = [
            f"TE_PREV_{column}_MEAN",
            f"TE_PREV_{column}_MAX",
            f"TE_PREV_{column}_COUNT",
        ]
        for output_column in output_columns:
            train_features[output_column] = np.nan

        if oof_train:
            for fit_pos, holdout_pos in inner_cv.split(train_index, y.loc[train_index]):
                fit_index = train_index[fit_pos]
                holdout_index = train_index[holdout_pos]
                mapping, global_mean = _fit_previous_mapping(
                    previous,
                    train_df,
                    y,
                    fit_index,
                    column,
                )
                encoded = _transform_previous_te(
                    previous,
                    train_df,
                    holdout_index,
                    mapping,
                    global_mean,
                    column,
                )
                train_features.loc[holdout_index, output_columns] = encoded[
                    output_columns
                ]
        else:
            mapping, global_mean = _fit_previous_mapping(
                previous,
                train_df,
                y,
                train_index,
                column,
            )
            encoded = _transform_previous_te(
                previous,
                train_df,
                train_index,
                mapping,
                global_mean,
                column,
            )
            train_features[output_columns] = encoded[output_columns]

        mapping, global_mean = _fit_previous_mapping(
            previous,
            train_df,
            y,
            train_index,
            column,
        )
        mappings["previous"][column] = {
            "mapping": mapping.to_dict(),
            "global_mean": global_mean,
        }
        transform_encoded = _transform_previous_te(
            previous,
            train_df,
            transform_index,
            mapping,
            global_mean,
            column,
        )
        transform_features[output_columns] = transform_encoded[output_columns]

    return train_features, transform_features, mappings


def build_model_frames(
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    train_df: pd.DataFrame,
    previous: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    transform_index: np.ndarray,
    oof_train: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_te, transform_te, mappings = build_target_encoding_frames(
        train_df=train_df,
        previous=previous,
        y=y,
        train_index=train_index,
        transform_index=transform_index,
        oof_train=oof_train,
    )
    X_train = pd.concat(
        [reduced.loc[train_index], recent_domain.loc[train_index], train_te],
        axis=1,
    )
    X_transform = pd.concat(
        [
            reduced.loc[transform_index],
            recent_domain.loc[transform_index],
            transform_te,
        ],
        axis=1,
    )

    return X_train, X_transform, mappings


def fit_transformed_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[LGBMClassifier, object, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = make_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_eval_transformed = preprocessor.transform(X_eval)
    transformed_names = np.asarray(preprocessor.get_feature_names_out())
    selected_mask = np.asarray(
        [name not in DROPPED_TRANSFORMED_FEATURES for name in transformed_names]
    )
    model = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
    model.fit(X_train_transformed[:, selected_mask], y_train)

    return (
        model,
        preprocessor,
        selected_mask,
        transformed_names,
        X_eval_transformed[:, selected_mask],
    )


def write_report(
    final_metrics: pd.DataFrame,
    previous_metrics: pd.DataFrame,
    confusion: np.ndarray,
    classification: str,
    threshold: float,
    transformed_feature_count: int,
) -> None:
    previous_selected = previous_metrics[
        previous_metrics["threshold_strategy"] == "validation_selected"
    ].copy()
    comparison = pd.concat(
        [
            previous_selected.assign(model="final_reduced_lightgbm_optuna")[
                [
                    "model",
                    "threshold",
                    "roc_auc",
                    "average_precision",
                    "accuracy",
                    "precision_class_1",
                    "recall_class_1",
                    "f1_class_1",
                ]
            ],
            final_metrics[
                final_metrics["threshold_strategy"] == "validation_selected"
            ][
                [
                    "model",
                    "threshold",
                    "roc_auc",
                    "average_precision",
                    "accuracy",
                    "precision_class_1",
                    "recall_class_1",
                    "f1_class_1",
                ]
            ],
        ],
        ignore_index=True,
    )
    lines = [
        "# Final Advanced Pruned Holdout Report",
        "",
        "Run date: 2026-07-16",
        "",
        "## Setup",
        "",
        "- Same train/holdout split as the reduced-model experiments.",
        "- Final feature recipe: top-200 reduced base + recent/domain relational features + fold-safe target encodings.",
        "- Dropped transformed features:",
        "  - `categorical__OCCUPATION_TYPE_infrequent_sklearn`",
        "  - `categorical__OCCUPATION_TYPE_Security staff`",
        f"- Selected transformed feature count: `{transformed_feature_count}`",
        f"- Threshold selected on validation split: `{threshold:.6f}`",
        "",
        "## Holdout Metrics",
        "",
        as_markdown(final_metrics),
        "",
        "## Comparison Against Previous Saved Reduced Optuna Model",
        "",
        as_markdown(comparison),
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
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

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

    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous = load_previous_te_rows()

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
    validation_model, validation_preprocessor, validation_mask, validation_names, X_valid_selected = (
        fit_transformed_model(
            X_train_full,
            y.loc[train_full_index],
            X_valid,
        )
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
    final_model, final_preprocessor, final_mask, final_names, X_test_selected = (
        fit_transformed_model(
            X_train_final,
            y.loc[train_index],
            X_test,
        )
    )
    test_probabilities = final_model.predict_proba(X_test_selected)[:, 1]
    final_metrics = pd.DataFrame(
        [
            {
                "model": "final_advanced_pruned_lgbm",
                "threshold_strategy": "default_0.5",
                **evaluate_probabilities(y.loc[test_index], test_probabilities, 0.5),
            },
            {
                "model": "final_advanced_pruned_lgbm",
                "threshold_strategy": "validation_selected",
                **evaluate_probabilities(
                    y.loc[test_index],
                    test_probabilities,
                    selected_threshold,
                ),
            },
        ]
    )
    final_metrics.to_csv(
        REPORTS_DIR / "final_advanced_pruned_holdout_metrics.csv",
        index=False,
    )

    selected_predictions = (test_probabilities >= selected_threshold).astype(int)
    confusion = confusion_matrix(y.loc[test_index], selected_predictions)
    classification = classification_report(
        y.loc[test_index],
        selected_predictions,
        zero_division=0,
    )
    pd.DataFrame(
        confusion,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(REPORTS_DIR / "final_advanced_pruned_holdout_confusion_matrix.csv")
    (REPORTS_DIR / "final_advanced_pruned_holdout_classification_report.txt").write_text(
        classification,
        encoding="utf-8",
    )

    selected_transformed_features = final_names[final_mask]
    pd.DataFrame({"transformed_feature": selected_transformed_features}).to_csv(
        REPORTS_DIR / "final_advanced_pruned_transformed_features.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": final_model,
            "preprocessor": final_preprocessor,
            "selected_mask": final_mask,
            "transformed_feature_names": final_names,
            "selected_transformed_features": selected_transformed_features,
            "dropped_transformed_features": sorted(DROPPED_TRANSFORMED_FEATURES),
            "raw_feature_columns": X_train_final.columns.tolist(),
            "target_encoding_mappings": target_encoding_mappings,
            "threshold": selected_threshold,
            "lightgbm_params": OPTUNA_LIGHTGBM_PARAMS,
        },
        MODEL_PATH,
        compress=3,
    )

    previous_metrics = pd.read_csv(
        REPORTS_DIR / "reduced_model_selection_test_metrics.csv"
    )
    write_report(
        final_metrics=final_metrics,
        previous_metrics=previous_metrics,
        confusion=confusion,
        classification=classification,
        threshold=selected_threshold,
        transformed_feature_count=int(final_mask.sum()),
    )

    print(final_metrics.to_string(index=False))
    print(confusion)
    print(f"Saved {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
