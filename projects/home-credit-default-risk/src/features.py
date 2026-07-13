import numpy as np
import pandas as pd


BASIC_APPLICATION_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


ENGINEERED_FEATURE_GROUPS = {
    "time": [
        "AGE_YEARS",
        "EMPLOYED_YEARS",
        "EMPLOYMENT_AGE_RATIO",
        "REGISTRATION_YEARS",
        "ID_PUBLISH_YEARS",
        "PHONE_CHANGE_YEARS",
        "REGISTRATION_AGE_RATIO",
        "ID_PUBLISH_AGE_RATIO",
        "PHONE_CHANGE_AGE_RATIO",
    ],
    "financial": [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "GOODS_CREDIT_RATIO",
        "CREDIT_TERM_APPROX",
    ],
    "household": [
        "INCOME_PER_PERSON",
        "CHILDREN_RATIO",
        "CREDIT_PER_PERSON",
        "ANNUITY_PER_PERSON",
    ],
    "external": [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_MAX",
        "EXT_SOURCE_STD",
        "EXT_SOURCE_COUNT",
    ],
    "flags": [
        "CONTACT_FLAGS_SUM",
        "ADDRESS_MISMATCH_COUNT",
        "DOCUMENT_COUNT",
    ],
    "social": [
        "SOCIAL_DEFAULT_RATIO_30",
        "SOCIAL_DEFAULT_RATIO_60",
        "SOCIAL_DEFAULT_TOTAL",
    ],
    "bureau_requests": [
        "BUREAU_REQUEST_TOTAL",
        "BUREAU_REQUEST_RECENT",
        "BUREAU_REQUEST_RECENT_TO_YEAR_RATIO",
    ],
    "region_timing": [
        "REGION_RATING_DIFF",
        "IS_WEEKEND_APPLICATION",
        "IS_NIGHT_APPLICATION",
    ],
    "building": [
        "BUILDING_FEATURE_MEAN",
        "BUILDING_FEATURE_MISSING_COUNT",
        "BUILDING_FEATURE_AVAILABLE_COUNT",
    ],
}


def _has_columns(data: pd.DataFrame, columns: list[str]) -> bool:
    return set(columns).issubset(data.columns)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, row-level features.

    The function is intentionally tolerant of missing source columns so smaller
    feature subsets used in earlier notebooks can still be rerun.
    """

    data = data.copy()

    if "DAYS_BIRTH" in data.columns:
        data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25

    if "DAYS_EMPLOYED" in data.columns:
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)
        data["EMPLOYED_YEARS"] = -data["DAYS_EMPLOYED"] / 365.25

    if _has_columns(data, ["EMPLOYED_YEARS", "AGE_YEARS"]):
        data["EMPLOYMENT_AGE_RATIO"] = _safe_divide(
            data["EMPLOYED_YEARS"], data["AGE_YEARS"]
        )

    if "DAYS_REGISTRATION" in data.columns:
        data["REGISTRATION_YEARS"] = -data["DAYS_REGISTRATION"] / 365.25

    if "DAYS_ID_PUBLISH" in data.columns:
        data["ID_PUBLISH_YEARS"] = -data["DAYS_ID_PUBLISH"] / 365.25

    if "DAYS_LAST_PHONE_CHANGE" in data.columns:
        data["PHONE_CHANGE_YEARS"] = -data["DAYS_LAST_PHONE_CHANGE"] / 365.25

    if _has_columns(data, ["REGISTRATION_YEARS", "AGE_YEARS"]):
        data["REGISTRATION_AGE_RATIO"] = _safe_divide(
            data["REGISTRATION_YEARS"], data["AGE_YEARS"]
        )

    if _has_columns(data, ["ID_PUBLISH_YEARS", "AGE_YEARS"]):
        data["ID_PUBLISH_AGE_RATIO"] = _safe_divide(
            data["ID_PUBLISH_YEARS"], data["AGE_YEARS"]
        )

    if _has_columns(data, ["PHONE_CHANGE_YEARS", "AGE_YEARS"]):
        data["PHONE_CHANGE_AGE_RATIO"] = _safe_divide(
            data["PHONE_CHANGE_YEARS"], data["AGE_YEARS"]
        )

    if _has_columns(data, ["AMT_CREDIT", "AMT_INCOME_TOTAL"]):
        data["CREDIT_INCOME_RATIO"] = _safe_divide(
            data["AMT_CREDIT"], data["AMT_INCOME_TOTAL"]
        )

    if _has_columns(data, ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]):
        data["ANNUITY_INCOME_RATIO"] = _safe_divide(
            data["AMT_ANNUITY"], data["AMT_INCOME_TOTAL"]
        )

    if _has_columns(data, ["AMT_GOODS_PRICE", "AMT_CREDIT"]):
        data["GOODS_CREDIT_RATIO"] = _safe_divide(
            data["AMT_GOODS_PRICE"], data["AMT_CREDIT"]
        )

    if _has_columns(data, ["AMT_CREDIT", "AMT_ANNUITY"]):
        data["CREDIT_TERM_APPROX"] = _safe_divide(data["AMT_CREDIT"], data["AMT_ANNUITY"])

    if _has_columns(data, ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]):
        data["INCOME_PER_PERSON"] = _safe_divide(
            data["AMT_INCOME_TOTAL"], data["CNT_FAM_MEMBERS"]
        )

    if _has_columns(data, ["CNT_CHILDREN", "CNT_FAM_MEMBERS"]):
        data["CHILDREN_RATIO"] = _safe_divide(
            data["CNT_CHILDREN"], data["CNT_FAM_MEMBERS"]
        )

    if _has_columns(data, ["AMT_CREDIT", "CNT_FAM_MEMBERS"]):
        data["CREDIT_PER_PERSON"] = _safe_divide(data["AMT_CREDIT"], data["CNT_FAM_MEMBERS"])

    if _has_columns(data, ["AMT_ANNUITY", "CNT_FAM_MEMBERS"]):
        data["ANNUITY_PER_PERSON"] = _safe_divide(
            data["AMT_ANNUITY"], data["CNT_FAM_MEMBERS"]
        )

    external_columns = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    if _has_columns(data, external_columns):
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

    if _has_columns(data, contact_columns):
        data["CONTACT_FLAGS_SUM"] = data[contact_columns].sum(axis=1)

    address_columns = [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]

    if _has_columns(data, address_columns):
        data["ADDRESS_MISMATCH_COUNT"] = data[address_columns].sum(axis=1)

    if _has_columns(data, ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]):
        data["REGION_RATING_DIFF"] = (
            data["REGION_RATING_CLIENT"] - data["REGION_RATING_CLIENT_W_CITY"]
        )

    if "WEEKDAY_APPR_PROCESS_START" in data.columns:
        data["IS_WEEKEND_APPLICATION"] = data["WEEKDAY_APPR_PROCESS_START"].isin(
            ["SATURDAY", "SUNDAY"]
        ).astype(int)

    if "HOUR_APPR_PROCESS_START" in data.columns:
        data["IS_NIGHT_APPLICATION"] = data["HOUR_APPR_PROCESS_START"].isin(
            [0, 1, 2, 3, 4, 5, 22, 23]
        ).astype(int)

    document_columns = [
        column for column in data.columns if column.startswith("FLAG_DOCUMENT_")
    ]

    if document_columns:
        data["DOCUMENT_COUNT"] = data[document_columns].sum(axis=1)

    if _has_columns(data, ["DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"]):
        data["SOCIAL_DEFAULT_RATIO_30"] = _safe_divide(
            data["DEF_30_CNT_SOCIAL_CIRCLE"], data["OBS_30_CNT_SOCIAL_CIRCLE"]
        )

    if _has_columns(data, ["DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"]):
        data["SOCIAL_DEFAULT_RATIO_60"] = _safe_divide(
            data["DEF_60_CNT_SOCIAL_CIRCLE"], data["OBS_60_CNT_SOCIAL_CIRCLE"]
        )

    if _has_columns(data, ["DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]):
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

    if _has_columns(data, bureau_request_columns):
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

        data["BUREAU_REQUEST_RECENT_TO_YEAR_RATIO"] = _safe_divide(
            data["BUREAU_REQUEST_RECENT"], data["AMT_REQ_CREDIT_BUREAU_YEAR"]
        )

    building_numeric_columns = [
        column
        for column in data.columns
        if column.endswith(("_AVG", "_MODE", "_MEDI"))
        and pd.api.types.is_numeric_dtype(data[column])
    ]

    if building_numeric_columns:
        data["BUILDING_FEATURE_MEAN"] = data[building_numeric_columns].mean(axis=1)

        data["BUILDING_FEATURE_MISSING_COUNT"] = (
            data[building_numeric_columns].isna().sum(axis=1)
        )

        data["BUILDING_FEATURE_AVAILABLE_COUNT"] = (
            data[building_numeric_columns].notna().sum(axis=1)
        )

    data = data.drop(columns=["DAYS_BIRTH"], errors="ignore")

    return data
