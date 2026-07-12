import numpy as np
import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, row-level features."""

    data = data.copy()

    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25

    data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

    data["EMPLOYED_YEARS"] = -data["DAYS_EMPLOYED"] / 365.25

    data["EMPLOYMENT_AGE_RATIO"] = data["EMPLOYED_YEARS"] / data["AGE_YEARS"].replace(
        0, np.nan
    )

    data["REGISTRATION_YEARS"] = -data["DAYS_REGISTRATION"] / 365.25

    data["ID_PUBLISH_YEARS"] = -data["DAYS_ID_PUBLISH"] / 365.25

    data["PHONE_CHANGE_YEARS"] = -data["DAYS_LAST_PHONE_CHANGE"] / 365.25

    data["CREDIT_INCOME_RATIO"] = data["AMT_CREDIT"] / data["AMT_INCOME_TOTAL"].replace(
        0, np.nan
    )

    data["ANNUITY_INCOME_RATIO"] = data["AMT_ANNUITY"] / data[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)

    data["GOODS_CREDIT_RATIO"] = data["AMT_GOODS_PRICE"] / data["AMT_CREDIT"].replace(
        0, np.nan
    )

    data["CREDIT_TERM_APPROX"] = data["AMT_CREDIT"] / data["AMT_ANNUITY"].replace(
        0, np.nan
    )

    data["INCOME_PER_PERSON"] = data["AMT_INCOME_TOTAL"] / data[
        "CNT_FAM_MEMBERS"
    ].replace(0, np.nan)

    data["CHILDREN_RATIO"] = data["CNT_CHILDREN"] / data["CNT_FAM_MEMBERS"].replace(
        0, np.nan
    )

    data["CREDIT_PER_PERSON"] = data["AMT_CREDIT"] / data["CNT_FAM_MEMBERS"].replace(
        0, np.nan
    )

    data["ANNUITY_PER_PERSON"] = data["AMT_ANNUITY"] / data["CNT_FAM_MEMBERS"].replace(
        0, np.nan
    )

    external_columns = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    data["EXT_SOURCE_MEAN"] = data[external_columns].mean(axis=1)

    data["EXT_SOURCE_MIN"] = data[external_columns].min(axis=1)

    data["EXT_SOURCE_MAX"] = data[external_columns].max(axis=1)

    data["EXT_SOURCE_STD"] = data[external_columns].std(axis=1, ddof=0)

    data["EXT_SOURCE_COUNT"] = data[external_columns].notna().sum(axis=1)

    contact_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]

    data["CONTACT_FLAGS_SUM"] = data[contact_columns].sum(axis=1)

    address_columns = [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]

    data["ADDRESS_MISMATCH_COUNT"] = data[address_columns].sum(axis=1)

    document_columns = [
        column for column in data.columns if column.startswith("FLAG_DOCUMENT_")
    ]

    data["DOCUMENT_COUNT"] = data[document_columns].sum(axis=1)

    data["SOCIAL_DEFAULT_RATIO_30"] = data["DEF_30_CNT_SOCIAL_CIRCLE"] / data[
        "OBS_30_CNT_SOCIAL_CIRCLE"
    ].replace(0, np.nan)

    data["SOCIAL_DEFAULT_RATIO_60"] = data["DEF_60_CNT_SOCIAL_CIRCLE"] / data[
        "OBS_60_CNT_SOCIAL_CIRCLE"
    ].replace(0, np.nan)

    data["SOCIAL_DEFAULT_TOTAL"] = (
        data["DEF_30_CNT_SOCIAL_CIRCLE"] + data["DEF_60_CNT_SOCIAL_CIRCLE"]
    )

    bureau_request_columns = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    data["BUREAU_REQUEST_TOTAL"] = data[bureau_request_columns].sum(
        axis=1,
        min_count=1,
    )

    data["BUREAU_REQUEST_RECENT"] = data[
        [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
        ]
    ].sum(axis=1, min_count=1)

    building_numeric_columns = [
        column
        for column in data.columns
        if column.endswith(("_AVG", "_MODE", "_MEDI"))
        and pd.api.types.is_numeric_dtype(data[column])
    ]

    data["BUILDING_FEATURE_MEAN"] = data[building_numeric_columns].mean(axis=1)

    data["BUILDING_FEATURE_MISSING_COUNT"] = (
        data[building_numeric_columns].isna().sum(axis=1)
    )

    data["BUILDING_FEATURE_AVAILABLE_COUNT"] = (
        data[building_numeric_columns].notna().sum(axis=1)
    )

    data = data.drop(columns=["DAYS_BIRTH"])

    return data
