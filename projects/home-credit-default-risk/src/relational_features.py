from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL_DAY_VALUE = 365243


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _flatten_columns(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = data.copy()
    data.columns = [
        f"{prefix}_{column}_{stat}".upper()
        for column, stat in data.columns.to_flat_index()
    ]
    return data


def _aggregate_numeric(
    data: pd.DataFrame,
    group_column: str,
    columns: list[str],
    prefix: str,
    aggregations: tuple[str, ...] = ("mean", "max", "min", "sum"),
) -> pd.DataFrame:
    existing_columns = [column for column in columns if column in data.columns]

    if not existing_columns:
        return pd.DataFrame(index=pd.Index([], name=group_column))

    grouped = data.groupby(group_column)[existing_columns].agg(list(aggregations))

    return _flatten_columns(grouped, prefix)


def _aggregate_dummies(
    data: pd.DataFrame,
    group_column: str,
    column: str,
    prefix: str,
) -> pd.DataFrame:
    if column not in data.columns:
        return pd.DataFrame(index=pd.Index([], name=group_column))

    dummies = pd.get_dummies(
        data[column],
        prefix=prefix,
        dummy_na=True,
        dtype=np.uint8,
    )
    dummies[group_column] = data[group_column].to_numpy()
    counts = dummies.groupby(group_column).sum()
    counts.columns = [column.upper() for column in counts.columns]

    totals = counts.sum(axis=1).replace(0, np.nan)
    rates = counts.div(totals, axis=0)
    rates.columns = [f"{column}_RATE" for column in counts.columns]

    return pd.concat([counts, rates], axis=1)


def build_bureau_features(
    bureau_path: str | Path,
    bureau_balance_path: str | Path | None = None,
) -> pd.DataFrame:
    bureau_columns = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_CURRENCY",
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
    ]

    bureau = pd.read_csv(bureau_path, usecols=bureau_columns)

    bureau["BURO_DEBT_CREDIT_RATIO"] = _safe_divide(
        bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"]
    )
    bureau["BURO_OVERDUE_CREDIT_RATIO"] = _safe_divide(
        bureau["AMT_CREDIT_SUM_OVERDUE"], bureau["AMT_CREDIT_SUM"]
    )
    bureau["BURO_ENDDATE_CREDIT_GAP"] = (
        bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    )

    numeric_features = [
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "DAYS_CREDIT_UPDATE",
        "AMT_ANNUITY",
        "BURO_DEBT_CREDIT_RATIO",
        "BURO_OVERDUE_CREDIT_RATIO",
        "BURO_ENDDATE_CREDIT_GAP",
    ]

    features = [
        bureau.groupby("SK_ID_CURR").size().rename("BURO_RECORD_COUNT").to_frame(),
        _aggregate_numeric(bureau, "SK_ID_CURR", numeric_features, "BURO"),
        _aggregate_dummies(bureau, "SK_ID_CURR", "CREDIT_ACTIVE", "BURO_ACTIVE"),
        _aggregate_dummies(bureau, "SK_ID_CURR", "CREDIT_TYPE", "BURO_TYPE"),
    ]

    currency_nunique = bureau.groupby("SK_ID_CURR")["CREDIT_CURRENCY"].nunique()
    credit_type_nunique = bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique()
    features.append(currency_nunique.rename("BURO_CREDIT_CURRENCY_NUNIQUE").to_frame())
    features.append(credit_type_nunique.rename("BURO_CREDIT_TYPE_NUNIQUE").to_frame())

    if bureau_balance_path is not None and Path(bureau_balance_path).exists():
        balance = pd.read_csv(bureau_balance_path)

        balance_status = _aggregate_dummies(
            balance,
            "SK_ID_BUREAU",
            "STATUS",
            "BB_STATUS",
        )
        balance_numeric = _aggregate_numeric(
            balance,
            "SK_ID_BUREAU",
            ["MONTHS_BALANCE"],
            "BB",
            aggregations=("mean", "max", "min", "count"),
        )
        balance_by_bureau = pd.concat([balance_numeric, balance_status], axis=1)
        balance_by_bureau["BB_BAD_STATUS_COUNT"] = balance_by_bureau[
            [
                column
                for column in balance_by_bureau.columns
                if column in {
                    "BB_STATUS_1",
                    "BB_STATUS_2",
                    "BB_STATUS_3",
                    "BB_STATUS_4",
                    "BB_STATUS_5",
                }
            ]
        ].sum(axis=1, min_count=1)
        balance_by_bureau = balance_by_bureau.reset_index()

        balance_with_curr = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(
            balance_by_bureau,
            on="SK_ID_BUREAU",
            how="left",
        )
        balance_features = _aggregate_numeric(
            balance_with_curr.drop(columns=["SK_ID_BUREAU"]),
            "SK_ID_CURR",
            [column for column in balance_with_curr.columns if column.startswith("BB_")],
            "BURO_BAL",
            aggregations=("mean", "max", "sum"),
        )
        features.append(balance_features)

    result = pd.concat(features, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_previous_application_features(path: str | Path) -> pd.DataFrame:
    columns = [
        "SK_ID_CURR",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "HOUR_APPR_PROCESS_START",
        "NFLAG_LAST_APPL_IN_DAY",
        "RATE_DOWN_PAYMENT",
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED",
        "DAYS_DECISION",
        "SELLERPLACE_AREA",
        "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
        "NFLAG_INSURED_ON_APPROVAL",
        "NAME_CONTRACT_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "FLAG_LAST_APPL_PER_CONTRACT",
        "NAME_CASH_LOAN_PURPOSE",
        "NAME_CONTRACT_STATUS",
        "NAME_PAYMENT_TYPE",
        "CODE_REJECT_REASON",
        "NAME_TYPE_SUITE",
        "NAME_CLIENT_TYPE",
        "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE",
        "CHANNEL_TYPE",
        "NAME_SELLER_INDUSTRY",
        "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION",
    ]

    previous = pd.read_csv(path, usecols=columns)

    day_columns = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    previous[day_columns] = previous[day_columns].replace(SENTINEL_DAY_VALUE, np.nan)

    previous["PREV_CREDIT_APPLICATION_RATIO"] = _safe_divide(
        previous["AMT_CREDIT"], previous["AMT_APPLICATION"]
    )
    previous["PREV_DOWN_PAYMENT_APPLICATION_RATIO"] = _safe_divide(
        previous["AMT_DOWN_PAYMENT"], previous["AMT_APPLICATION"]
    )
    previous["PREV_ANNUITY_CREDIT_RATIO"] = _safe_divide(
        previous["AMT_ANNUITY"], previous["AMT_CREDIT"]
    )

    numeric_features = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "HOUR_APPR_PROCESS_START",
        "NFLAG_LAST_APPL_IN_DAY",
        "RATE_DOWN_PAYMENT",
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED",
        "DAYS_DECISION",
        "SELLERPLACE_AREA",
        "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
        "NFLAG_INSURED_ON_APPROVAL",
        "PREV_CREDIT_APPLICATION_RATIO",
        "PREV_DOWN_PAYMENT_APPLICATION_RATIO",
        "PREV_ANNUITY_CREDIT_RATIO",
    ]

    categorical_columns = [
        "NAME_CONTRACT_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
        "FLAG_LAST_APPL_PER_CONTRACT",
        "NAME_CASH_LOAN_PURPOSE",
        "NAME_CONTRACT_STATUS",
        "NAME_PAYMENT_TYPE",
        "CODE_REJECT_REASON",
        "NAME_TYPE_SUITE",
        "NAME_CLIENT_TYPE",
        "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE",
        "CHANNEL_TYPE",
        "NAME_SELLER_INDUSTRY",
        "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION",
    ]

    features = [
        previous.groupby("SK_ID_CURR").size().rename("PREV_RECORD_COUNT").to_frame(),
        _aggregate_numeric(previous, "SK_ID_CURR", numeric_features, "PREV"),
    ]
    features.extend(
        _aggregate_dummies(previous, "SK_ID_CURR", column, f"PREV_{column}")
        for column in categorical_columns
    )

    result = pd.concat(features, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_installments_features(path: str | Path) -> pd.DataFrame:
    columns = [
        "SK_ID_CURR",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]

    installments = pd.read_csv(path, usecols=columns)
    installments["INST_PAYMENT_DELAY"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )
    installments["INST_PAYMENT_RATIO"] = _safe_divide(
        installments["AMT_PAYMENT"], installments["AMT_INSTALMENT"]
    )
    installments["INST_PAYMENT_DIFF"] = (
        installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    )
    installments["INST_LATE_PAYMENT_FLAG"] = (
        installments["INST_PAYMENT_DELAY"] > 0
    ).astype(np.uint8)
    installments["INST_UNDERPAYMENT_FLAG"] = (
        installments["INST_PAYMENT_DIFF"] > 0
    ).astype(np.uint8)

    numeric_features = [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "INST_PAYMENT_DELAY",
        "INST_PAYMENT_RATIO",
        "INST_PAYMENT_DIFF",
        "INST_LATE_PAYMENT_FLAG",
        "INST_UNDERPAYMENT_FLAG",
    ]

    features = [
        installments.groupby("SK_ID_CURR")
        .size()
        .rename("INST_RECORD_COUNT")
        .to_frame(),
        _aggregate_numeric(installments, "SK_ID_CURR", numeric_features, "INST"),
    ]

    result = pd.concat(features, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_pos_cash_features(path: str | Path) -> pd.DataFrame:
    columns = [
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]

    pos_cash = pd.read_csv(path, usecols=columns)
    pos_cash["POS_COMPLETION_RATIO"] = 1 - _safe_divide(
        pos_cash["CNT_INSTALMENT_FUTURE"], pos_cash["CNT_INSTALMENT"]
    )

    numeric_features = [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
        "POS_COMPLETION_RATIO",
    ]

    features = [
        pos_cash.groupby("SK_ID_CURR").size().rename("POS_RECORD_COUNT").to_frame(),
        _aggregate_numeric(pos_cash, "SK_ID_CURR", numeric_features, "POS"),
        _aggregate_dummies(
            pos_cash,
            "SK_ID_CURR",
            "NAME_CONTRACT_STATUS",
            "POS_STATUS",
        ),
    ]

    result = pd.concat(features, axis=1)
    result.index.name = "SK_ID_CURR"

    return result


def build_credit_card_features(path: str | Path) -> pd.DataFrame:
    columns = [
        "SK_ID_CURR",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_DRAWINGS_OTHER_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL",
        "AMT_RECIVABLE",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_ATM_CURRENT",
        "CNT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_OTHER_CURRENT",
        "CNT_DRAWINGS_POS_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]

    credit_card = pd.read_csv(path, usecols=columns)
    credit_card["CC_UTILIZATION_RATIO"] = _safe_divide(
        credit_card["AMT_BALANCE"], credit_card["AMT_CREDIT_LIMIT_ACTUAL"]
    )
    credit_card["CC_PAYMENT_MIN_RATIO"] = _safe_divide(
        credit_card["AMT_PAYMENT_TOTAL_CURRENT"],
        credit_card["AMT_INST_MIN_REGULARITY"],
    )
    credit_card["CC_DRAWING_LIMIT_RATIO"] = _safe_divide(
        credit_card["AMT_DRAWINGS_CURRENT"],
        credit_card["AMT_CREDIT_LIMIT_ACTUAL"],
    )

    numeric_features = [
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_DRAWINGS_OTHER_CURRENT",
        "AMT_DRAWINGS_POS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_RECEIVABLE_PRINCIPAL",
        "AMT_RECIVABLE",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_ATM_CURRENT",
        "CNT_DRAWINGS_CURRENT",
        "CNT_DRAWINGS_OTHER_CURRENT",
        "CNT_DRAWINGS_POS_CURRENT",
        "CNT_INSTALMENT_MATURE_CUM",
        "SK_DPD",
        "SK_DPD_DEF",
        "CC_UTILIZATION_RATIO",
        "CC_PAYMENT_MIN_RATIO",
        "CC_DRAWING_LIMIT_RATIO",
    ]

    features = [
        credit_card.groupby("SK_ID_CURR").size().rename("CC_RECORD_COUNT").to_frame(),
        _aggregate_numeric(credit_card, "SK_ID_CURR", numeric_features, "CC"),
        _aggregate_dummies(
            credit_card,
            "SK_ID_CURR",
            "NAME_CONTRACT_STATUS",
            "CC_STATUS",
        ),
    ]

    result = pd.concat(features, axis=1)
    result.index.name = "SK_ID_CURR"

    return result
