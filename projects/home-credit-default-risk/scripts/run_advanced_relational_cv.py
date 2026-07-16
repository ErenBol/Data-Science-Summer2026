from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_relational_feature_experiments import (
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    align_to_application,
    build_application_features,
    build_feature_matrix,
    get_positive_probabilities,
)
from scripts.run_relational_feature_pruning import (
    RELATIONAL_GROUPS,
    as_markdown,
    build_lgbm_pipeline,
    top_gain_columns,
)


N_SPLITS = 3
INNER_TE_SPLITS = 3

OPTUNA_LIGHTGBM_PARAMS = {
    "n_estimators": 1008,
    "learning_rate": 0.02178196621205854,
    "num_leaves": 32,
    "max_depth": 9,
    "min_child_samples": 179,
    "subsample": 0.9978522016969403,
    "colsample_bytree": 0.8265635267137263,
    "reg_alpha": 1.224310860385835e-08,
    "reg_lambda": 2.6072505559148544e-06,
    "min_split_gain": 0.514970414020943,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
    "importance_type": "gain",
}

APPLICATION_TE_COLUMNS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
]

PREVIOUS_TE_COLUMNS = [
    "PRODUCT_COMBINATION",
    "NAME_GOODS_CATEGORY",
    "NAME_CASH_LOAN_PURPOSE",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _align_features(train_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    aligned = train_df[["SK_ID_CURR"]].merge(
        features,
        left_on="SK_ID_CURR",
        right_index=True,
        how="left",
    )
    aligned = aligned.drop(columns=["SK_ID_CURR"])
    aligned.index = train_df.index

    return aligned


def _sum_boolean(grouped: pd.core.groupby.DataFrameGroupBy, column: str) -> pd.Series:
    return grouped[column].sum()


def build_reduced_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    source_gain = pd.read_csv(REPORTS_DIR / "relational_source_column_gain.csv")
    reduced_columns = top_gain_columns(source_gain, 200)

    app_features = build_application_features(train_df)
    app_features = app_features[
        [column for column in reduced_columns if column in app_features.columns]
    ]

    aligned_groups = {}
    for group_name in RELATIONAL_GROUPS:
        features = pd.read_pickle(PROCESSED_DIR / f"{group_name}_features.pkl")
        selected_columns = [
            column for column in reduced_columns if column in features.columns
        ]
        features = features[selected_columns]
        aligned_groups[group_name] = align_to_application(train_df, features)
        del features
        gc.collect()

    reduced = build_feature_matrix(app_features, aligned_groups, RELATIONAL_GROUPS)
    reduced = reduced.replace([np.inf, -np.inf], np.nan)

    return reduced[[column for column in reduced_columns if column in reduced.columns]]


def build_bureau_recent_domain_features() -> pd.DataFrame:
    bureau = pd.read_csv(
        RAW_DATA_DIR / "bureau.csv",
        usecols=[
            "SK_ID_CURR",
            "CREDIT_ACTIVE",
            "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE",
        ],
    )
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.uint8)
    bureau["IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(np.uint8)
    bureau["IS_OVERDUE"] = (bureau["CREDIT_DAY_OVERDUE"].fillna(0) > 0).astype(np.uint8)

    frames = []
    grouped = bureau.groupby("SK_ID_CURR")
    active = bureau[bureau["IS_ACTIVE"] == 1].groupby("SK_ID_CURR")
    closed = bureau[bureau["IS_CLOSED"] == 1].groupby("SK_ID_CURR")

    base = pd.DataFrame(index=grouped.size().index)
    base["ADV_BURO_ACTIVE_RECORD_COUNT"] = active.size()
    base["ADV_BURO_CLOSED_RECORD_COUNT"] = closed.size()
    base["ADV_BURO_ACTIVE_DEBT_SUM"] = active["AMT_CREDIT_SUM_DEBT"].sum()
    base["ADV_BURO_ACTIVE_CREDIT_SUM"] = active["AMT_CREDIT_SUM"].sum()
    base["ADV_BURO_ACTIVE_DEBT_CREDIT_RATIO"] = _safe_divide(
        base["ADV_BURO_ACTIVE_DEBT_SUM"],
        base["ADV_BURO_ACTIVE_CREDIT_SUM"],
    )
    base["ADV_BURO_CLOSED_DEBT_SUM"] = closed["AMT_CREDIT_SUM_DEBT"].sum()
    base["ADV_BURO_CLOSED_CREDIT_SUM"] = closed["AMT_CREDIT_SUM"].sum()
    base["ADV_BURO_CLOSED_DEBT_CREDIT_RATIO"] = _safe_divide(
        base["ADV_BURO_CLOSED_DEBT_SUM"],
        base["ADV_BURO_CLOSED_CREDIT_SUM"],
    )
    base["ADV_BURO_ACTIVE_MAX_OVERDUE"] = active["AMT_CREDIT_MAX_OVERDUE"].max()
    base["ADV_BURO_ACTIVE_MAX_DAYS_OVERDUE"] = active["CREDIT_DAY_OVERDUE"].max()
    frames.append(base)

    for months, days in [(6, 180), (12, 365), (24, 730)]:
        recent = bureau[bureau["DAYS_CREDIT"] >= -days].copy()
        recent_grouped = recent.groupby("SK_ID_CURR")
        recent_features = pd.DataFrame(index=recent_grouped.size().index)
        prefix = f"ADV_BURO_LAST_{months}M"
        recent_features[f"{prefix}_RECORD_COUNT"] = recent_grouped.size()
        recent_features[f"{prefix}_ACTIVE_COUNT"] = recent_grouped["IS_ACTIVE"].sum()
        recent_features[f"{prefix}_ACTIVE_RATIO"] = _safe_divide(
            recent_features[f"{prefix}_ACTIVE_COUNT"],
            recent_features[f"{prefix}_RECORD_COUNT"],
        )
        recent_features[f"{prefix}_OVERDUE_COUNT"] = recent_grouped["IS_OVERDUE"].sum()
        recent_features[f"{prefix}_OVERDUE_RATIO"] = _safe_divide(
            recent_features[f"{prefix}_OVERDUE_COUNT"],
            recent_features[f"{prefix}_RECORD_COUNT"],
        )
        recent_features[f"{prefix}_DEBT_SUM"] = recent_grouped[
            "AMT_CREDIT_SUM_DEBT"
        ].sum()
        recent_features[f"{prefix}_CREDIT_SUM"] = recent_grouped[
            "AMT_CREDIT_SUM"
        ].sum()
        recent_features[f"{prefix}_DEBT_CREDIT_RATIO"] = _safe_divide(
            recent_features[f"{prefix}_DEBT_SUM"],
            recent_features[f"{prefix}_CREDIT_SUM"],
        )
        recent_features[f"{prefix}_MAX_OVERDUE"] = recent_grouped[
            "AMT_CREDIT_SUM_OVERDUE"
        ].max()
        frames.append(recent_features)

    result = pd.concat(frames, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_previous_recent_domain_features() -> pd.DataFrame:
    previous = pd.read_csv(
        RAW_DATA_DIR / "previous_application.csv",
        usecols=[
            "SK_ID_CURR",
            "DAYS_DECISION",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "CNT_PAYMENT",
        ],
    )
    previous["IS_REFUSED"] = (
        previous["NAME_CONTRACT_STATUS"] == "Refused"
    ).astype(np.uint8)
    previous["IS_APPROVED"] = (
        previous["NAME_CONTRACT_STATUS"] == "Approved"
    ).astype(np.uint8)
    previous["PREV_CREDIT_APPLICATION_RATIO"] = _safe_divide(
        previous["AMT_CREDIT"],
        previous["AMT_APPLICATION"],
    )

    frames = []
    for months, days in [(6, 180), (12, 365), (24, 730)]:
        recent = previous[previous["DAYS_DECISION"] >= -days].copy()
        grouped = recent.groupby("SK_ID_CURR")
        features = pd.DataFrame(index=grouped.size().index)
        prefix = f"ADV_PREV_LAST_{months}M"
        features[f"{prefix}_RECORD_COUNT"] = grouped.size()
        features[f"{prefix}_REFUSED_COUNT"] = grouped["IS_REFUSED"].sum()
        features[f"{prefix}_APPROVED_COUNT"] = grouped["IS_APPROVED"].sum()
        features[f"{prefix}_REFUSED_RATE"] = _safe_divide(
            features[f"{prefix}_REFUSED_COUNT"],
            features[f"{prefix}_RECORD_COUNT"],
        )
        features[f"{prefix}_CREDIT_APPLICATION_RATIO_MEAN"] = grouped[
            "PREV_CREDIT_APPLICATION_RATIO"
        ].mean()
        features[f"{prefix}_CNT_PAYMENT_MEAN"] = grouped["CNT_PAYMENT"].mean()
        frames.append(features)

    result = pd.concat(frames, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_installment_recent_domain_features() -> pd.DataFrame:
    installments = pd.read_csv(
        RAW_DATA_DIR / "installments_payments.csv",
        usecols=[
            "SK_ID_CURR",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
    )
    installments["PAYMENT_DELAY"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )
    installments["LATE_PAYMENT"] = (installments["PAYMENT_DELAY"] > 0).astype(np.uint8)
    installments["LATE_DELAY_ONLY"] = installments["PAYMENT_DELAY"].where(
        installments["PAYMENT_DELAY"] > 0,
        np.nan,
    )
    installments["UNDERPAYMENT_AMOUNT"] = (
        installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    ).clip(lower=0)
    installments["UNDERPAYMENT"] = (
        installments["UNDERPAYMENT_AMOUNT"] > 0
    ).astype(np.uint8)

    frames = []
    for months, days in [(6, 180), (12, 365), (24, 730)]:
        recent = installments[installments["DAYS_INSTALMENT"] >= -days].copy()
        grouped = recent.groupby("SK_ID_CURR")
        features = pd.DataFrame(index=grouped.size().index)
        prefix = f"ADV_INST_LAST_{months}M"
        features[f"{prefix}_RECORD_COUNT"] = grouped.size()
        features[f"{prefix}_LATE_RATE"] = grouped["LATE_PAYMENT"].mean()
        features[f"{prefix}_LATE_COUNT"] = grouped["LATE_PAYMENT"].sum()
        features[f"{prefix}_LATE_DELAY_MEAN_ONLY"] = grouped["LATE_DELAY_ONLY"].mean()
        features[f"{prefix}_LATE_DELAY_MAX"] = grouped["PAYMENT_DELAY"].max()
        features[f"{prefix}_UNDERPAYMENT_RATE"] = grouped["UNDERPAYMENT"].mean()
        features[f"{prefix}_UNDERPAYMENT_AMOUNT_SUM"] = grouped[
            "UNDERPAYMENT_AMOUNT"
        ].sum()
        frames.append(features)

    result = pd.concat(frames, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_credit_card_recent_domain_features() -> pd.DataFrame:
    credit_card = pd.read_csv(
        RAW_DATA_DIR / "credit_card_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "SK_DPD",
            "SK_DPD_DEF",
        ],
    )
    credit_card["UTILIZATION"] = _safe_divide(
        credit_card["AMT_BALANCE"],
        credit_card["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    credit_card["HAS_DPD"] = (credit_card["SK_DPD"].fillna(0) > 0).astype(np.uint8)
    credit_card["HAS_DPD_DEF"] = (
        credit_card["SK_DPD_DEF"].fillna(0) > 0
    ).astype(np.uint8)

    frames = []
    for months in [6, 12, 24]:
        recent = credit_card[credit_card["MONTHS_BALANCE"] >= -months].copy()
        grouped = recent.groupby("SK_ID_CURR")
        features = pd.DataFrame(index=grouped.size().index)
        prefix = f"ADV_CC_LAST_{months}M"
        features[f"{prefix}_RECORD_COUNT"] = grouped.size()
        features[f"{prefix}_UTILIZATION_MEAN"] = grouped["UTILIZATION"].mean()
        features[f"{prefix}_UTILIZATION_MAX"] = grouped["UTILIZATION"].max()
        features[f"{prefix}_DPD_RATE"] = grouped["HAS_DPD"].mean()
        features[f"{prefix}_DPD_DEF_RATE"] = grouped["HAS_DPD_DEF"].mean()
        features[f"{prefix}_DPD_MAX"] = grouped["SK_DPD"].max()
        frames.append(features)

    last6 = credit_card[credit_card["MONTHS_BALANCE"] >= -6].groupby("SK_ID_CURR")[
        "UTILIZATION"
    ].mean()
    prev6 = credit_card[
        (credit_card["MONTHS_BALANCE"] < -6)
        & (credit_card["MONTHS_BALANCE"] >= -12)
    ].groupby("SK_ID_CURR")["UTILIZATION"].mean()
    trend = (last6 - prev6).rename("ADV_CC_UTILIZATION_LAST6_MINUS_PREV6")
    frames.append(trend.to_frame())

    result = pd.concat(frames, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_pos_recent_domain_features() -> pd.DataFrame:
    pos = pd.read_csv(
        RAW_DATA_DIR / "POS_CASH_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "SK_DPD",
            "SK_DPD_DEF",
            "NAME_CONTRACT_STATUS",
        ],
    )
    pos["HAS_DPD"] = (pos["SK_DPD"].fillna(0) > 0).astype(np.uint8)
    pos["HAS_DPD_DEF"] = (pos["SK_DPD_DEF"].fillna(0) > 0).astype(np.uint8)
    pos["IS_COMPLETED"] = (
        pos["NAME_CONTRACT_STATUS"] == "Completed"
    ).astype(np.uint8)

    frames = []
    for months in [6, 12, 24]:
        recent = pos[pos["MONTHS_BALANCE"] >= -months].copy()
        grouped = recent.groupby("SK_ID_CURR")
        features = pd.DataFrame(index=grouped.size().index)
        prefix = f"ADV_POS_LAST_{months}M"
        features[f"{prefix}_RECORD_COUNT"] = grouped.size()
        features[f"{prefix}_DPD_RATE"] = grouped["HAS_DPD"].mean()
        features[f"{prefix}_DPD_DEF_RATE"] = grouped["HAS_DPD_DEF"].mean()
        features[f"{prefix}_DPD_MAX"] = grouped["SK_DPD"].max()
        features[f"{prefix}_COMPLETED_RATE"] = grouped["IS_COMPLETED"].mean()
        frames.append(features)

    result = pd.concat(frames, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_recent_domain_features(train_df: pd.DataFrame) -> pd.DataFrame:
    cache_path = PROCESSED_DIR / "advanced_recent_domain_features.pkl"
    if cache_path.exists():
        features = pd.read_pickle(cache_path)
    else:
        frames = [
            build_bureau_recent_domain_features(),
            build_previous_recent_domain_features(),
            build_installment_recent_domain_features(),
            build_credit_card_recent_domain_features(),
            build_pos_recent_domain_features(),
        ]
        features = pd.concat(frames, axis=1)
        features = features.replace([np.inf, -np.inf], np.nan)
        features.to_pickle(cache_path)

    return _align_features(train_df, features)


def load_previous_te_rows() -> pd.DataFrame:
    previous = pd.read_csv(
        RAW_DATA_DIR / "previous_application.csv",
        usecols=["SK_ID_CURR", *PREVIOUS_TE_COLUMNS],
    )

    return previous


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
    y: pd.Series,
    fit_index: np.ndarray,
    transform_index: np.ndarray,
    column: str,
) -> pd.Series:
    mapping, global_mean = _fit_category_mapping(
        train_df.loc[fit_index, column],
        y.loc[fit_index],
    )
    encoded = (
        train_df.loc[transform_index, column]
        .map(mapping)
        .fillna(global_mean)
        .rename(f"TE_APP_{column}")
    )

    return encoded


def _transform_previous_te(
    previous: pd.DataFrame,
    train_df: pd.DataFrame,
    y: pd.Series,
    fit_index: np.ndarray,
    transform_index: np.ndarray,
    column: str,
) -> pd.DataFrame:
    fit_targets = train_df.loc[fit_index, ["SK_ID_CURR"]].copy()
    fit_targets["TARGET"] = y.loc[fit_index].to_numpy()

    fit_previous = previous[["SK_ID_CURR", column]].merge(
        fit_targets,
        on="SK_ID_CURR",
        how="inner",
    )
    mapping, global_mean = _fit_category_mapping(
        fit_previous[column],
        fit_previous["TARGET"],
    )

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


def make_target_encoding_features(
    train_df: pd.DataFrame,
    previous: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = pd.DataFrame(index=train_index)
    valid_features = pd.DataFrame(index=valid_index)

    inner_cv = StratifiedKFold(
        n_splits=INNER_TE_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for column in APPLICATION_TE_COLUMNS:
        train_features[f"TE_APP_{column}"] = np.nan
        for inner_fit_pos, inner_holdout_pos in inner_cv.split(
            train_index,
            y.loc[train_index],
        ):
            inner_fit_index = train_index[inner_fit_pos]
            inner_holdout_index = train_index[inner_holdout_pos]
            train_features.loc[
                inner_holdout_index,
                f"TE_APP_{column}",
            ] = _transform_application_te(
                train_df,
                y,
                inner_fit_index,
                inner_holdout_index,
                column,
            )

        valid_features[f"TE_APP_{column}"] = _transform_application_te(
            train_df,
            y,
            train_index,
            valid_index,
            column,
        )

    for column in PREVIOUS_TE_COLUMNS:
        column_names = [
            f"TE_PREV_{column}_MEAN",
            f"TE_PREV_{column}_MAX",
            f"TE_PREV_{column}_COUNT",
        ]
        for name in column_names:
            train_features[name] = np.nan

        for inner_fit_pos, inner_holdout_pos in inner_cv.split(
            train_index,
            y.loc[train_index],
        ):
            inner_fit_index = train_index[inner_fit_pos]
            inner_holdout_index = train_index[inner_holdout_pos]
            encoded = _transform_previous_te(
                previous,
                train_df,
                y,
                inner_fit_index,
                inner_holdout_index,
                column,
            )
            train_features.loc[inner_holdout_index, column_names] = encoded[
                column_names
            ]

        valid_encoded = _transform_previous_te(
            previous,
            train_df,
            y,
            train_index,
            valid_index,
            column,
        )
        valid_features[column_names] = valid_encoded[column_names]

    return train_features, valid_features


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    pipeline = build_lgbm_pipeline(
        X_train=X_train,
        numeric_add_indicator=False,
        one_hot_min_frequency=1000,
        one_hot_max_categories=15,
    )
    pipeline.named_steps["model"].set_params(**OPTUNA_LIGHTGBM_PARAMS)

    return pipeline


def evaluate_fold(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    fold: int,
) -> dict[str, float | int | str]:
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    probabilities = get_positive_probabilities(pipeline, X_valid)

    return {
        "feature_set": name,
        "fold": fold,
        "raw_feature_count": X_train.shape[1],
        "transformed_feature_count": len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        ),
        "roc_auc": roc_auc_score(y_valid, probabilities),
        "average_precision": average_precision_score(y_valid, probabilities),
    }


def write_report(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="feature_set", columns="metric", values="value")
    lines = [
        "# Advanced Relational Feature CV Report",
        "",
        "Run date: 2026-07-16",
        "",
        "## Purpose",
        "",
        "This experiment tests recent-history relational aggregates, smarter domain "
        "aggregates, and fold-safe target/risk encodings with the latest Optuna "
        "LightGBM parameters.",
        "",
        "Validation uses `StratifiedKFold(n_splits=3)`. Target encodings are fitted "
        "inside each fold; training-fold target encodings are generated with an "
        "inner 3-fold split to avoid row-level leakage.",
        "",
        "## Fold Results",
        "",
        as_markdown(results),
        "",
        "## Summary",
        "",
        as_markdown(pivot.reset_index()),
        "",
    ]
    (REPORTS_DIR / "advanced_relational_cv_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    print("Reading application data")
    train_df = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    y = train_df["TARGET"].copy()

    print("Building reduced top-200 baseline matrix")
    reduced = build_reduced_matrix(train_df)

    print("Building recent/domain relational features")
    recent_domain = build_recent_domain_features(train_df)

    print("Loading previous-application rows for fold-safe target encodings")
    previous_te = load_previous_te_rows()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_pos, valid_pos) in enumerate(cv.split(reduced, y), start=1):
        train_index = reduced.index.to_numpy()[train_pos]
        valid_index = reduced.index.to_numpy()[valid_pos]
        print(f"Fold {fold}: baseline")

        rows.append(
            evaluate_fold(
                name="reduced_top_200",
                X_train=reduced.loc[train_index],
                y_train=y.loc[train_index],
                X_valid=reduced.loc[valid_index],
                y_valid=y.loc[valid_index],
                fold=fold,
            )
        )

        print(f"Fold {fold}: target encodings")
        train_te, valid_te = make_target_encoding_features(
            train_df=train_df,
            previous=previous_te,
            y=y,
            train_index=train_index,
            valid_index=valid_index,
        )

        X_train_advanced = pd.concat(
            [
                reduced.loc[train_index],
                recent_domain.loc[train_index],
                train_te,
            ],
            axis=1,
        )
        X_valid_advanced = pd.concat(
            [
                reduced.loc[valid_index],
                recent_domain.loc[valid_index],
                valid_te,
            ],
            axis=1,
        )

        rows.append(
            evaluate_fold(
                name="reduced_top_200_plus_advanced",
                X_train=X_train_advanced,
                y_train=y.loc[train_index],
                X_valid=X_valid_advanced,
                y_valid=y.loc[valid_index],
                fold=fold,
            )
        )

        pd.DataFrame(rows).to_csv(
            REPORTS_DIR / "advanced_relational_cv_results.csv",
            index=False,
        )
        gc.collect()

    results = pd.DataFrame(rows)
    summary_rows = []
    for feature_set, group in results.groupby("feature_set"):
        for metric in ["roc_auc", "average_precision", "raw_feature_count", "transformed_feature_count"]:
            summary_rows.append(
                {
                    "feature_set": feature_set,
                    "metric": f"{metric}_mean",
                    "value": group[metric].mean(),
                }
            )
            if metric in {"roc_auc", "average_precision"}:
                summary_rows.append(
                    {
                        "feature_set": feature_set,
                        "metric": f"{metric}_std",
                        "value": group[metric].std(ddof=0),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS_DIR / "advanced_relational_cv_summary.csv", index=False)
    write_report(results, summary)

    print(results.to_string(index=False))
    print(summary.pivot(index="feature_set", columns="metric", values="value"))


if __name__ == "__main__":
    main()
