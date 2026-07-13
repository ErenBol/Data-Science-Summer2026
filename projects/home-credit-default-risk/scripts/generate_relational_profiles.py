from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

try:
    from ydata_profiling import ProfileReport
except ImportError:
    from pandas_profiling import ProfileReport


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROFILE_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]


def build_profile(csv_name: str, sample_rows: int, force: bool) -> None:
    stem = Path(csv_name).stem
    html_path = REPORTS_DIR / f"{stem}_profile.html"
    json_path = REPORTS_DIR / f"{stem}_profile.json"

    if not force and html_path.exists() and json_path.exists():
        print(f"Skipping existing profile: {csv_name}")
        return

    csv_path = RAW_DATA_DIR / csv_name
    print(f"Reading {csv_path.name} with nrows={sample_rows:,}")
    data = pd.read_csv(csv_path, nrows=sample_rows)

    profile = ProfileReport(
        data,
        title=f"Home Credit {stem} Profiling Report",
        minimal=True,
        explorative=True,
    )

    print(f"Writing {html_path.name}")
    profile.to_file(html_path)
    print(f"Writing {json_path.name}")
    profile.to_file(json_path)


def main() -> None:
    parser = ArgumentParser(
        description="Generate ydata/pandas profile HTML and JSON reports."
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=200_000,
        help="Deterministic first-row sample size for large source CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate profiles even when HTML and JSON files already exist.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)

    for csv_name in PROFILE_FILES:
        build_profile(
            csv_name=csv_name,
            sample_rows=args.sample_rows,
            force=args.force,
        )


if __name__ == "__main__":
    main()
