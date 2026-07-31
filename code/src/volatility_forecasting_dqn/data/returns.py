"""Clean quote bars and construct five-minute intraday log returns."""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "Date",
    "Time",
    "Open Bid Price",
    "Open Ask Price",
    "Close Bid Price",
    "Close Ask Price",
]

BAR_MINUTES = 5
MAX_MISSING_RATIO = 0.05
ACTIVITY_BLOCK_BARS = 12

MORNING_OPEN = time(9, 30)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
AFTERNOON_CLOSE = time(15, 0)


def _bar_close_times(session_open: time, session_close: time) -> list[time]:
    current = datetime.combine(date(2000, 1, 1), session_open) + timedelta(
        minutes=BAR_MINUTES
    )
    end = datetime.combine(date(2000, 1, 1), session_close)
    result = []
    while current <= end:
        result.append(current.time())
        current += timedelta(minutes=BAR_MINUTES)
    return result


EXPECTED_TIMES = (
    _bar_close_times(MORNING_OPEN, MORNING_CLOSE)
    + _bar_close_times(AFTERNOON_OPEN, AFTERNOON_CLOSE)
)
EXPECTED_BARS_PER_DAY = len(EXPECTED_TIMES)  # 48 returns
REQUIRED_PRICES_PER_DAY = EXPECTED_BARS_PER_DAY + 2  # 48 closes + 2 opens
ACTIVITY_BLOCKS_PER_DAY = EXPECTED_BARS_PER_DAY // ACTIVITY_BLOCK_BARS
SESSION_START_TIMES = {EXPECTED_TIMES[0], EXPECTED_TIMES[24]}


def calculate_intraday_returns(quotes: pd.DataFrame) -> pd.DataFrame:
    """Convert one stock's quote bars into session-local log returns.

    A valid day contains 48 five-minute returns. Their 50 required midpoint
    prices are the 48 bar closes and the first opening price of each session.
    The morning and afternoon sessions are calculated separately, so lunch and
    overnight price changes are excluded.
    """

    missing_columns = sorted(set(RAW_COLUMNS) - set(quotes.columns))
    if missing_columns:
        raise ValueError(
            "Missing raw quote columns: " + ", ".join(missing_columns)
        )

    df = quotes[RAW_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.normalize()
    df["Time"] = pd.to_datetime(
        df["Time"].astype(str),
        format="%H:%M:%S",
        errors="raise",
    ).dt.time

    quote_columns = RAW_COLUMNS[2:]
    df[quote_columns] = df[quote_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    df[quote_columns] = df[quote_columns].mask(df[quote_columns].eq(0))

    df["Open"] = (df["Open Bid Price"] + df["Open Ask Price"]) / 2
    df["Close"] = (df["Close Bid Price"] + df["Close Ask Price"]) / 2
    df = df.loc[df["Time"].isin(EXPECTED_TIMES)].copy()
    df = df.sort_values(["Date", "Time"], kind="stable")

    if df.empty:
        return pd.DataFrame(columns=["Date", "Time", "log_return"])

    df["session"] = np.where(df["Time"] <= MORNING_CLOSE, "AM", "PM")
    is_session_start = df["Time"].isin(SESSION_START_TIMES)

    # Count missing values among the 50 prices required by the 48 returns.
    df["required_price_missing"] = (
        df["Close"].isna().astype(int)
        + (is_session_start & df["Open"].isna()).astype(int)
    )
    missing_count = df.groupby("Date")["required_price_missing"].transform(
        "sum"
    )
    df["missing_ratio"] = missing_count / REQUIRED_PRICES_PER_DAY

    # Require the complete 48-bar grid and at most 5% missing required prices.
    valid_grid_dates = []
    for trading_date, day in df.groupby("Date", sort=False):
        if tuple(sorted(day["Time"])) == tuple(EXPECTED_TIMES):
            valid_grid_dates.append(trading_date)
    df = df.loc[
        df["Date"].isin(valid_grid_dates)
        & df["missing_ratio"].le(MAX_MISSING_RATIO)
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["Date", "Time", "log_return"])

    # Preserve the filling rule used by the original notebook.
    df["Close_filled"] = df["Close"].fillna(df["Open"])
    df["Close_filled"] = df.groupby(["Date", "session"])[
        "Close_filled"
    ].ffill()
    df["Close_filled"] = df.groupby(["Date", "session"])[
        "Close_filled"
    ].bfill()
    df["Open_filled"] = df["Open"].fillna(df["Close_filled"])

    previous_close = df.groupby(["Date", "session"])["Close_filled"].shift()
    denominator = previous_close.fillna(df["Open_filled"])
    df["log_return"] = np.log(df["Close_filled"] / denominator)

    # Remove a day if any 12-bar (one-hour) block has zero realized variation.
    df["bar_number"] = df.groupby("Date").cumcount()
    df["activity_block"] = df["bar_number"] // ACTIVITY_BLOCK_BARS
    block_rv = df.groupby(["Date", "activity_block"])[
        "log_return"
    ].apply(lambda values: values.pow(2).sum())
    active_dates = block_rv.groupby(level="Date").filter(
        lambda values: (
            len(values) == ACTIVITY_BLOCKS_PER_DAY
            and bool(values.gt(0).all())
        )
    ).index.get_level_values("Date").unique()

    result = df.loc[
        df["Date"].isin(active_dates),
        ["Date", "Time", "log_return"],
    ].copy()
    return result.sort_values(["Date", "Time"]).reset_index(drop=True)


def clean_quote_file(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Clean one raw quote CSV and write its return CSV."""

    result = calculate_intraday_returns(pd.read_csv(input_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def clean_quote_directory(
    raw_directory: str | Path,
    processed_directory: str | Path,
) -> list[Path]:
    """Clean all CSV files in a directory."""

    raw_dir = Path(raw_directory)
    output_dir = Path(processed_directory)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written_files = []
    for input_path in files:
        output_path = output_dir / input_path.name
        clean_quote_file(input_path, output_path)
        written_files.append(output_path)
    return written_files
