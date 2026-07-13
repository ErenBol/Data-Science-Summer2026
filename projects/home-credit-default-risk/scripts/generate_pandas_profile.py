from pathlib import Path

import pandas as pd

try:
    from ydata_profiling import ProfileReport
except ImportError:
    from pandas_profiling import ProfileReport


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(
        RAW_DATA_DIR / "application_train.csv",
    )

    profile = ProfileReport(
        train_df,
        title="Home Credit Application Train Profiling Report",
        minimal=True,
        explorative=True,
    )

    profile.to_file(
        REPORTS_DIR / "application_train_profile.html",
    )


if __name__ == "__main__":
    main()
