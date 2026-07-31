"""Realized-volatility features built from five-minute log returns."""

import numpy as np
import pandas as pd

from volatility_forecasting_dqn.data.returns import (
    ACTIVITY_BLOCK_BARS,
    EXPECTED_BARS_PER_DAY,
    EXPECTED_TIMES,
)


def calculate_realized_volatility(
    returns: pd.DataFrame,
    frequency: str = "1D",
) -> pd.DataFrame:
    """Calculate daily or hourly RV and log-RV.

    Parameters
    ----------
    returns:
        Output of ``calculate_intraday_returns`` with columns
        ``Date``, ``Time`` and ``log_return``.
    frequency:
        ``"1D"`` sums all 48 intraday squared returns.
        ``"1H"`` sums each consecutive block of 12 squared returns.
    """

    if frequency not in {"1D", "1H"}:
        raise ValueError("frequency must be '1D' or '1H'")

    required_columns = {"Date", "Time", "log_return"}
    missing_columns = sorted(required_columns - set(returns.columns))
    if missing_columns:
        raise ValueError(
            "Missing return columns: " + ", ".join(missing_columns)
        )

    df = returns[["Date", "Time", "log_return"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.normalize()
    df["Time"] = pd.to_datetime(
        df["Time"].astype(str),
        format="%H:%M:%S",
        errors="raise",
    ).dt.time
    df["log_return"] = pd.to_numeric(df["log_return"], errors="raise")
    df = df.sort_values(["Date", "Time"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("returns is empty")
    if not np.isfinite(df["log_return"]).all():
        raise ValueError("log_return contains non-finite values")

    expected_times = tuple(EXPECTED_TIMES)
    for trading_date, day in df.groupby("Date", sort=False):
        if tuple(day["Time"]) != expected_times:
            raise ValueError(
                f"{trading_date.date()} does not contain the expected "
                f"{EXPECTED_BARS_PER_DAY} five-minute returns"
            )

    df["squared_return"] = df["log_return"].pow(2)

    if frequency == "1D":
        result = (
            df.groupby("Date", as_index=False)
            .agg(Time=("Time", "max"), rv=("squared_return", "sum"))
        )
    else:
        df["hour_block"] = df.groupby("Date").cumcount() // ACTIVITY_BLOCK_BARS
        result = (
            df.groupby(["Date", "hour_block"], as_index=False)
            .agg(Time=("Time", "max"), rv=("squared_return", "sum"))
            .drop(columns="hour_block")
        )

    if result["rv"].le(0).any():
        raise ValueError("RV must be positive before taking logarithms")

    result["log_rv"] = np.log(result["rv"])
    return result[["Date", "Time", "rv", "log_rv"]]
