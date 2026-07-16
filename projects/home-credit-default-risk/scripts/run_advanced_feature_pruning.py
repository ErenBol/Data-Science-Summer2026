from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_advanced_relational_cv import (
    N_SPLITS,
    OPTUNA_LIGHTGBM_PARAMS,
    build_recent_domain_features,
    build_reduced_matrix,
    load_previous_te_rows,
    make_target_encoding_features,
)
from scripts.run_relational_feature_experiments import (
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from scripts.run_relational_feature_pruning import as_markdown
from src.preprocessing import build_preprocessor


MAX_ITERATIONS = 20
ACCEPTANCE_TOLERANCE = 0.00015
PRACTICAL_TOLERANCE = 0.00050
DROP_SEQUENCE = [1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150]


def build_advanced_fold_frames(
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    train_df: pd.DataFrame,
    previous_te: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_te, valid_te = make_target_encoding_features(
        train_df=train_df,
        previous=previous_te,
        y=y,
        train_index=train_index,
        valid_index=valid_index,
    )

    X_train = pd.concat(
        [
            reduced.loc[train_index],
            recent_domain.loc[train_index],
            train_te,
        ],
        axis=1,
    )
    X_valid = pd.concat(
        [
            reduced.loc[valid_index],
            recent_domain.loc[valid_index],
            valid_te,
        ],
        axis=1,
    )

    return X_train, X_valid


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

    return build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_add_indicator=False,
        one_hot_sparse_output=True,
        one_hot_min_frequency=1000,
        one_hot_max_categories=15,
    )


def fit_eval_transformed_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    selected_feature_names: set[str] | None = None,
) -> dict:
    preprocessor = make_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_valid_transformed = preprocessor.transform(X_valid)
    feature_names = np.asarray(preprocessor.get_feature_names_out())

    if selected_feature_names is None:
        mask = np.ones(len(feature_names), dtype=bool)
    else:
        mask = np.asarray([name in selected_feature_names for name in feature_names])
        if not mask.any():
            raise ValueError("No transformed features remained after selection.")

    model = LGBMClassifier(**OPTUNA_LIGHTGBM_PARAMS)
    model.fit(X_train_transformed[:, mask], y_train)
    probabilities = model.predict_proba(X_valid_transformed[:, mask])[:, 1]
    selected_names = feature_names[mask]
    importance = pd.DataFrame(
        {
            "transformed_feature": selected_names,
            "gain_importance": model.feature_importances_,
        }
    ).sort_values("gain_importance", ascending=True)

    return {
        "roc_auc": roc_auc_score(y_valid, probabilities),
        "average_precision": average_precision_score(y_valid, probabilities),
        "feature_count": len(selected_names),
        "selected_names": set(selected_names),
        "importance": importance,
    }


def run_iterative_pruning_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    baseline = fit_eval_transformed_selection(X_train, y_train, X_valid, y_valid)
    all_names = baseline["selected_names"]
    weakest_features = baseline["importance"]["transformed_feature"].tolist()
    candidate_sets = {
        "advanced_all_transformed": all_names,
    }
    search_rows = [
        {
            "iteration": 0,
            "candidate_action": "baseline",
            "dropped_count": 0,
            "remaining_features": baseline["feature_count"],
            "roc_auc": baseline["roc_auc"],
            "average_precision": baseline["average_precision"],
            "accepted": True,
            "candidate_name": "advanced_all_transformed",
        }
    ]
    dropped_rows = []

    for iteration, drop_count in enumerate(DROP_SEQUENCE[:MAX_ITERATIONS], start=1):
        if drop_count >= len(weakest_features):
            break

        drop_batch = weakest_features[:drop_count]
        candidate_names = all_names - set(drop_batch)
        candidate_name = f"drop_{drop_count}_weakest_transformed"
        candidate = fit_eval_transformed_selection(
            X_train,
            y_train,
            X_valid,
            y_valid,
            candidate_names,
        )
        accepted = candidate["roc_auc"] >= baseline["roc_auc"] - PRACTICAL_TOLERANCE
        candidate_sets[candidate_name] = candidate_names

        search_rows.append(
            {
                "iteration": iteration,
                "candidate_action": "drop_cumulative_weakest_transformed_features",
                "dropped_count": len(drop_batch),
                "remaining_features": candidate["feature_count"],
                "roc_auc": candidate["roc_auc"],
                "average_precision": candidate["average_precision"],
                "accepted": accepted,
                "candidate_name": candidate_name,
            }
        )
        dropped_rows.extend(
            {
                "iteration": iteration,
                "transformed_feature": feature,
                "accepted": accepted,
            }
            for feature in drop_batch
        )

        gc.collect()

    return pd.DataFrame(search_rows), pd.DataFrame(dropped_rows), candidate_sets


def choose_cv_candidates(
    search_results: pd.DataFrame,
    candidate_sets: dict[str, set[str]],
) -> dict[str, set[str] | None]:
    candidates: dict[str, set[str] | None] = {
        "advanced_all_transformed": None,
    }
    non_baseline = search_results[search_results["iteration"] > 0].copy()

    if non_baseline.empty:
        return candidates

    best_row = non_baseline.sort_values(
        ["roc_auc", "average_precision"],
        ascending=False,
    ).iloc[0]
    candidates[f"advanced_{best_row['candidate_name']}"] = candidate_sets[
        best_row["candidate_name"]
    ]

    practical = non_baseline[non_baseline["accepted"]].sort_values(
        ["dropped_count", "roc_auc"],
        ascending=[False, False],
    )
    if not practical.empty:
        practical_row = practical.iloc[0]
        candidates[f"advanced_{practical_row['candidate_name']}"] = candidate_sets[
            practical_row["candidate_name"]
        ]

    ap_row = non_baseline.sort_values(
        ["average_precision", "roc_auc"],
        ascending=False,
    ).iloc[0]
    candidates[f"advanced_{ap_row['candidate_name']}"] = candidate_sets[
        ap_row["candidate_name"]
    ]

    return candidates


def evaluate_cv_with_selection(
    reduced: pd.DataFrame,
    recent_domain: pd.DataFrame,
    train_df: pd.DataFrame,
    previous_te: pd.DataFrame,
    y: pd.Series,
    selected_names: set[str] | None,
    feature_set_name: str,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
        train_index = reduced.index.to_numpy()[train_pos]
        valid_index = reduced.index.to_numpy()[valid_pos]
        X_train, X_valid = build_advanced_fold_frames(
            reduced,
            recent_domain,
            train_df,
            previous_te,
            y,
            train_index,
            valid_index,
        )
        result = fit_eval_transformed_selection(
            X_train,
            y.loc[train_index],
            X_valid,
            y.loc[valid_index],
            selected_names,
        )
        rows.append(
            {
                "feature_set": feature_set_name,
                "fold": fold,
                "transformed_feature_count": result["feature_count"],
                "roc_auc": result["roc_auc"],
                "average_precision": result["average_precision"],
            }
        )
        gc.collect()

    return pd.DataFrame(rows)


def summarize_cv(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, group in results.groupby("feature_set"):
        for metric in ["roc_auc", "average_precision", "transformed_feature_count"]:
            rows.append(
                {
                    "feature_set": feature_set,
                    "metric": f"{metric}_mean",
                    "value": group[metric].mean(),
                }
            )
            if metric in {"roc_auc", "average_precision"}:
                rows.append(
                    {
                        "feature_set": feature_set,
                        "metric": f"{metric}_std",
                        "value": group[metric].std(ddof=0),
                    }
                )

    return pd.DataFrame(rows)


def write_report(
    search_results: pd.DataFrame,
    dropped_features: pd.DataFrame,
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    selected_feature_set: str,
    selected_feature_count: int,
) -> None:
    summary_pivot = cv_summary.pivot(
        index="feature_set",
        columns="metric",
        values="value",
    ).reset_index()
    selected_candidate_name = selected_feature_set.removeprefix("advanced_")
    selected_rows = search_results[
        search_results["candidate_name"] == selected_candidate_name
    ]
    if selected_rows.empty:
        selected_drops = pd.DataFrame(
            columns=["iteration", "transformed_feature", "accepted"]
        )
    else:
        selected_iteration = int(selected_rows.iloc[0]["iteration"])
        selected_drops = dropped_features[
            dropped_features["iteration"] == selected_iteration
        ].copy()

    lines = [
        "# Advanced Feature Pruning Report",
        "",
        "Run date: 2026-07-16",
        "",
        "## Purpose",
        "",
        "This experiment prunes transformed features created by the preprocessing "
        "pipeline, not only raw input columns. One-hot encoded categories and "
        "numeric transformed features are evaluated by LightGBM gain importance.",
        "",
        "The pruning search uses the first stratified fold as a development fold. "
        "Each iteration drops the lowest-gain transformed feature batch and retrains. "
        f"The maximum number of pruning iterations is `{MAX_ITERATIONS}`.",
        "",
        "## Iterative Search",
        "",
        as_markdown(search_results),
        "",
        "## Dropped Features In Selected Set",
        "",
        as_markdown(selected_drops),
        "",
        f"Selected CV feature set: `{selected_feature_set}`",
        "",
        f"Selected transformed feature count after pruning: `{selected_feature_count}`",
        "",
        "## Final 3-Fold CV Comparison",
        "",
        as_markdown(cv_results),
        "",
        "## CV Summary",
        "",
        as_markdown(summary_pivot),
        "",
    ]
    (REPORTS_DIR / "advanced_feature_pruning_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    print("Reading application data")
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    print("Building reduced and advanced feature matrices")
    reduced = build_reduced_matrix(train_df)
    recent_domain = build_recent_domain_features(train_df)
    previous_te = load_previous_te_rows()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_pos, valid_pos = next(cv.split(reduced, y))
    train_index = reduced.index.to_numpy()[train_pos]
    valid_index = reduced.index.to_numpy()[valid_pos]

    print("Preparing development fold target encodings")
    X_train, X_valid = build_advanced_fold_frames(
        reduced,
        recent_domain,
        train_df,
        previous_te,
        y,
        train_index,
        valid_index,
    )

    print("Running transformed-feature pruning sequence")
    search_results, dropped_features, candidate_sets = run_iterative_pruning_search(
        X_train,
        y.loc[train_index],
        X_valid,
        y.loc[valid_index],
    )
    cv_candidates = choose_cv_candidates(search_results, candidate_sets)

    search_results.to_csv(
        REPORTS_DIR / "advanced_feature_pruning_search.csv",
        index=False,
    )
    dropped_features.to_csv(
        REPORTS_DIR / "advanced_feature_pruning_dropped_features.csv",
        index=False,
    )

    print("Running final 3-fold CV comparison")
    cv_frames = []
    for feature_set_name, selected_names in cv_candidates.items():
        cv_frames.append(
            evaluate_cv_with_selection(
                reduced,
                recent_domain,
                train_df,
                previous_te,
                y,
                selected_names=selected_names,
                feature_set_name=feature_set_name,
            )
        )

    cv_results = pd.concat(cv_frames, ignore_index=True)
    cv_summary = summarize_cv(cv_results)
    summary_pivot = cv_summary.pivot(
        index="feature_set",
        columns="metric",
        values="value",
    )
    selected_feature_set = summary_pivot.sort_values(
        ["roc_auc_mean", "average_precision_mean"],
        ascending=False,
    ).index[0]
    selected_names = cv_candidates[selected_feature_set]
    if selected_names is None:
        selected_names = candidate_sets["advanced_all_transformed"]
    selected_features = pd.DataFrame({"transformed_feature": sorted(selected_names)})
    selected_features.to_csv(
        REPORTS_DIR / "advanced_feature_pruning_selected_features.csv",
        index=False,
    )
    cv_results.to_csv(
        REPORTS_DIR / "advanced_feature_pruning_cv_results.csv",
        index=False,
    )
    cv_summary.to_csv(
        REPORTS_DIR / "advanced_feature_pruning_cv_summary.csv",
        index=False,
    )

    write_report(
        search_results=search_results,
        dropped_features=dropped_features,
        cv_results=cv_results,
        cv_summary=cv_summary,
        selected_feature_set=selected_feature_set,
        selected_feature_count=len(selected_names),
    )

    print(search_results.to_string(index=False))
    print(cv_summary.pivot(index="feature_set", columns="metric", values="value"))


if __name__ == "__main__":
    main()
