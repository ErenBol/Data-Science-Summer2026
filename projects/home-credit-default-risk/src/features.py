import numpy as np
import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, row-level features."""

    data = data.copy()

    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25

    data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

    data["CREDIT_INCOME_RATIO"] = data["AMT_CREDIT"] / data["AMT_INCOME_TOTAL"].replace(
        0, np.nan
    )

    data["ANNUITY_INCOME_RATIO"] = data["AMT_ANNUITY"] / data[
        "AMT_INCOME_TOTAL"
    ].replace(0, np.nan)

    data["CREDIT_TERM"] = data["AMT_ANNUITY"] / data["AMT_CREDIT"].replace(0, np.nan)

    data = data.drop(columns=["DAYS_BIRTH"])

    return data
