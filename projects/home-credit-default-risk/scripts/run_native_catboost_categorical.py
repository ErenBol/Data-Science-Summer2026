from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_blend import (
    CATBOOST_PARAMS,
    BLEND_WEIGHTS,
    blend_name,
    make_blend_probabilities,
    selected_transformed_matrices,
    summarize_blends,
)
from scripts.run_advanced_feature_pruning import build_advanced_fold_frames
from scripts.run_advanced_relational_cv import (
    N_SPLITS,
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
)
from scripts.run_relational_feature_pruning import as_markdown


CURRENT_CATBOOST_CV_ROC_AUC = 0.791913
CURRENT_BLEND_CV_ROC_AUC = 0.793581

REPORT_PATH = REPORTS_DIR / "native_catboost_categorical_report.md"
CATBOOST_CV_PATH = REPORTS_DIR / "native_catboost_categorical_cv_results.csv"
LIGHTGBM_CV_PATH = REPORTS_DIR / "native_catboost_blend_lightgbm_cv_results.csv"
BLEND_CV_PATH = REPORTS_DIR / "native_catboost_blend_cv_results.csv"
BLEND_SUMMARY_PATH = REPORTS_DIR / "native_catboost_blend_cv_summary.csv"
FEATURES_PATH = REPORTS_DIR / "native_catboost_categorical_features.csv"


def build_previous_raw_categorical_features() -> pd.DataFrame:
    previous = pd.read_csv(
        RAW_DATA_DIR / "previous_application.csv",
        usecols=["SK_ID_CURR", "DAYS_DECISION", *PREVIOUS_TE_COLUMNS],
    )
    previous = previous.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
    grouped = previous.groupby("SK_ID_CURR", sort=False)
    features = pd.DataFrame(index=grouped.size().index)

    for column in PREVIOUS_TE_COLUMNS:
        recent = grouped[column].first()
        mode = grouped[column].agg(
            lambda values: values.mode(dropna=True).iloc[0]
            if not values.mode(dropna=True).empty
            else np.nan
        )
        features[f"CAT_PREV_RECENT_{column}"] = recent
        features[f"CAT_PREV_MODE_{column}"] = mode

    features.index.name = "SK_ID_CURR"

    return features


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


def build_native_catboost_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_raw = align_to_application(train_df, build_previous_raw_categorical_features())
    matrix = pd.concat([reduced, recent_domain, previous_raw], axis=1)
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    return matrix


def prepare_native_catboost_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[str]]:
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

    categorical_indices = [result.columns.get_loc(column) for column in categorical_columns]

    return result, categorical_indices, categorical_columns


def fit_native_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[np.ndarray, int, list[str]]:
    X_train_cb, categorical_indices, categorical_columns = prepare_native_catboost_frame(
        X_train
    )
    X_valid_cb, _, _ = prepare_native_catboost_frame(X_valid)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        Pool(X_train_cb, y_train, cat_features=categorical_indices),
        eval_set=Pool(X_valid_cb, y_valid, cat_features=categorical_indices),
        use_best_model=True,
    )
    probabilities = model.predict_proba(X_valid_cb)[:, 1]

    return probabilities, int(model.best_iteration_ or CATBOOST_PARAMS["iterations"]), categorical_columns


def run_native_catboost_cv(
    X_native: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(X_native, y), start=1):
        train_index = X_native.index.to_numpy()[train_pos]
        valid_index = X_native.index.to_numpy()[valid_pos]
        print(f"Fold {fold}: native CatBoost")
        probabilities, best_iteration, categorical_columns = fit_native_catboost(
            X_native.loc[train_index],
            y.loc[train_index],
            X_native.loc[valid_index],
            y.loc[valid_index],
        )
        rows.append(
            {
                "model": "native_catboost_categorical",
                "fold": fold,
                "raw_feature_count": X_native.shape[1],
                "categorical_feature_count": len(categorical_columns),
                "best_iteration": best_iteration,
                "roc_auc": roc_auc_score(y.loc[valid_index], probabilities),
                "average_precision": average_precision_score(
                    y.loc[valid_index],
                    probabilities,
                ),
            }
        )
        pd.DataFrame(rows).to_csv(CATBOOST_CV_PATH, index=False)
        gc.collect()

    pd.DataFrame({"feature": X_native.columns}).to_csv(FEATURES_PATH, index=False)

    return pd.DataFrame(rows)


def run_blend_cv(
    X_native: pd.DataFrame,
    train_df: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lightgbm_rows = []
    blend_rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
        train_index = reduced.index.to_numpy()[train_pos]
        valid_index = reduced.index.to_numpy()[valid_pos]
        print(f"Fold {fold}: LightGBM side for native CatBoost blend")
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
        lightgbm_probabilities = lightgbm.predict_proba(X_lgbm_valid_selected)[:, 1]
        lightgbm_rows.append(
            {
                "model": "lightgbm_advanced_pruned",
                "fold": fold,
                "transformed_feature_count": int(selected_mask.sum()),
                "roc_auc": roc_auc_score(y.loc[valid_index], lightgbm_probabilities),
                "average_precision": average_precision_score(
                    y.loc[valid_index],
                    lightgbm_probabilities,
                ),
            }
        )

        print(f"Fold {fold}: native CatBoost side for blend")
        catboost_probabilities, best_iteration, categorical_columns = fit_native_catboost(
            X_native.loc[train_index],
            y.loc[train_index],
            X_native.loc[valid_index],
            y.loc[valid_index],
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
                    "catboost_best_iteration": best_iteration,
                    "catboost_categorical_feature_count": len(categorical_columns),
                    "roc_auc": roc_auc_score(y.loc[valid_index], probabilities),
                    "average_precision": average_precision_score(
                        y.loc[valid_index],
                        probabilities,
                    ),
                }
            )
        pd.DataFrame(lightgbm_rows).to_csv(LIGHTGBM_CV_PATH, index=False)
        pd.DataFrame(blend_rows).to_csv(BLEND_CV_PATH, index=False)
        del (
            X_lgbm_train,
            X_lgbm_valid,
            X_lgbm_train_selected,
            X_lgbm_valid_selected,
            lightgbm,
        )
        gc.collect()

    return pd.DataFrame(lightgbm_rows), pd.DataFrame(blend_rows)


def summarize_catboost(results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "previous_transformed_catboost",
                "mean_cv_roc_auc": CURRENT_CATBOOST_CV_ROC_AUC,
                "std_cv_roc_auc": np.nan,
                "mean_cv_average_precision": np.nan,
                "std_cv_average_precision": np.nan,
            },
            {
                "model": "native_catboost_categorical",
                "mean_cv_roc_auc": results["roc_auc"].mean(),
                "std_cv_roc_auc": results["roc_auc"].std(ddof=0),
                "mean_cv_average_precision": results["average_precision"].mean(),
                "std_cv_average_precision": results["average_precision"].std(ddof=0),
            },
        ]
    )


def write_report(
    catboost_cv: pd.DataFrame,
    catboost_summary: pd.DataFrame,
    lightgbm_cv: pd.DataFrame | None,
    blend_cv: pd.DataFrame | None,
    blend_summary: pd.DataFrame | None,
) -> None:
    native_row = catboost_summary[
        catboost_summary["model"] == "native_catboost_categorical"
    ].iloc[0]
    catboost_delta = pd.DataFrame(
        [
            {
                "metric": "mean_cv_roc_auc",
                "delta": native_row["mean_cv_roc_auc"] - CURRENT_CATBOOST_CV_ROC_AUC,
            }
        ]
    )
    lines = [
        "# Native CatBoost Categorical Report",
        "",
        "Run date: 2026-07-17",
        "",
        "## Setup",
        "",
        "- Trained CatBoost on the raw advanced feature frame instead of the LightGBM-style transformed matrix.",
        "- Raw application categorical columns and raw previous-application categorical replacements are passed through `cat_features`.",
        "- Fold-safe `TE_*` target-encoding columns are intentionally omitted from the native CatBoost frame.",
        "- Numeric top-200 reduced features and advanced recent/domain relational features are kept unchanged.",
        "- CV protocol: same 3-fold stratified split as `run_advanced_relational_cv.py`.",
        "",
        "## Native CatBoost CV Results",
        "",
        as_markdown(catboost_cv),
        "",
        "## CatBoost Comparison",
        "",
        as_markdown(catboost_summary),
        "",
        "Delta versus previous transformed CatBoost:",
        "",
        as_markdown(catboost_delta),
        "",
    ]

    if blend_cv is None or blend_summary is None:
        lines.extend(
            [
                f"Native CatBoost did not improve over the previous CatBoost CV ROC-AUC `{CURRENT_CATBOOST_CV_ROC_AUC:.6f}`, so the LightGBM+CatBoost blend search was not rerun.",
                "",
            ]
        )
    else:
        best_blend = blend_summary.iloc[0]
        lines.extend(
            [
                "Native CatBoost improved over the previous CatBoost result, so the LightGBM+CatBoost blend search was rerun.",
                "",
                "## LightGBM CV Side",
                "",
                as_markdown(lightgbm_cv),
                "",
                "## Blend CV Results",
                "",
                as_markdown(blend_cv),
                "",
                "## Blend CV Summary",
                "",
                as_markdown(blend_summary),
                "",
                f"Best native-CatBoost blend: `{best_blend['blend']}` with mean CV ROC-AUC `{best_blend['mean_cv_roc_auc']:.6f}`.",
                f"Current final advanced blend CV ROC-AUC baseline: `{CURRENT_BLEND_CV_ROC_AUC:.6f}`.",
                f"Delta: `{best_blend['mean_cv_roc_auc'] - CURRENT_BLEND_CV_ROC_AUC:.6f}`.",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    print("Reading application data")
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    print("Building native CatBoost feature matrix")
    X_native = build_native_catboost_matrix(train_df)

    print("Running native CatBoost CV")
    catboost_cv = run_native_catboost_cv(X_native, y)
    catboost_summary = summarize_catboost(catboost_cv)
    catboost_summary.to_csv(
        REPORTS_DIR / "native_catboost_categorical_cv_summary.csv",
        index=False,
    )

    lightgbm_cv = None
    blend_cv = None
    blend_summary = None
    native_auc = float(
        catboost_summary.loc[
            catboost_summary["model"] == "native_catboost_categorical",
            "mean_cv_roc_auc",
        ].iloc[0]
    )
    if native_auc > CURRENT_CATBOOST_CV_ROC_AUC:
        print("Native CatBoost improved; rerunning blend search")
        lightgbm_cv, blend_cv = run_blend_cv(X_native, train_df, y)
        blend_summary = summarize_blends(blend_cv)
        blend_summary.to_csv(BLEND_SUMMARY_PATH, index=False)
    else:
        print("Native CatBoost did not improve; skipping blend search")

    write_report(
        catboost_cv=catboost_cv,
        catboost_summary=catboost_summary,
        lightgbm_cv=lightgbm_cv,
        blend_cv=blend_cv,
        blend_summary=blend_summary,
    )

    print(catboost_cv.to_string(index=False))
    print(catboost_summary.to_string(index=False))
    if blend_summary is not None:
        print(blend_summary.to_string(index=False))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
