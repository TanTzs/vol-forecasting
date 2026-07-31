"""Core checks for the intraday return calculation."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from volatility_forecasting_dqn.data import (  # noqa: E402
    EXPECTED_BARS_PER_DAY,
    REQUIRED_PRICES_PER_DAY,
    calculate_intraday_returns,
    clean_quote_file,
)
from volatility_forecasting_dqn.data.returns import (  # noqa: E402
    ACTIVITY_BLOCK_BARS,
    EXPECTED_TIMES,
)


def make_day(
    trading_date: str,
    morning_price: float = 100.0,
    afternoon_price: float = 200.0,
    flat_block: int | None = None,
) -> pd.DataFrame:
    rows = []
    session_size = EXPECTED_BARS_PER_DAY // 2
    for session, start_price in enumerate([morning_price, afternoon_price]):
        price = start_price
        times = EXPECTED_TIMES[
            session * session_size : (session + 1) * session_size
        ]
        for local_index, bar_time in enumerate(times):
            bar_number = session * session_size + local_index
            open_price = price
            if (
                flat_block is not None
                and bar_number // ACTIVITY_BLOCK_BARS == flat_block
            ):
                close_price = open_price
            else:
                close_price = open_price * (
                    1 + 0.001 * ((bar_number % 3) + 1)
                )

            rows.append(
                {
                    "Date": trading_date,
                    "Time": bar_time.strftime("%H:%M:%S"),
                    "Open Bid Price": open_price - 0.005,
                    "Open Ask Price": open_price + 0.005,
                    "Close Bid Price": close_price - 0.005,
                    "Close Ask Price": close_price + 0.005,
                }
            )
            price = close_price
    return pd.DataFrame(rows)


class IntradayReturnTests(unittest.TestCase):
    def test_produces_48_returns_from_50_required_prices(self) -> None:
        self.assertEqual(EXPECTED_BARS_PER_DAY, 48)
        self.assertEqual(REQUIRED_PRICES_PER_DAY, 50)

        result = calculate_intraday_returns(make_day("2024-01-02"))

        self.assertEqual(len(result), 48)
        self.assertEqual(
            result.columns.tolist(),
            ["Date", "Time", "log_return"],
        )
        self.assertTrue(np.isfinite(result["log_return"]).all())

    def test_allows_two_missing_required_prices_but_rejects_three(self) -> None:
        accepted = make_day("2024-01-02")
        rejected = make_day("2024-01-03")
        for row in [3, 17]:
            accepted.loc[
                row,
                ["Close Bid Price", "Close Ask Price"],
            ] = 0
        for row in [3, 17, 31]:
            rejected.loc[
                row,
                ["Close Bid Price", "Close Ask Price"],
            ] = 0

        result = calculate_intraday_returns(
            pd.concat([accepted, rejected], ignore_index=True)
        )

        dates = result["Date"].dt.strftime("%Y-%m-%d").unique().tolist()
        self.assertEqual(dates, ["2024-01-02"])

    def test_excludes_lunch_and_overnight_price_changes(self) -> None:
        day_one = make_day(
            "2024-01-02",
            morning_price=100,
            afternoon_price=1_000,
        )
        day_two = make_day(
            "2024-01-03",
            morning_price=10_000,
            afternoon_price=100_000,
        )

        result = calculate_intraday_returns(
            pd.concat([day_one, day_two], ignore_index=True)
        )
        first_bar_times = [EXPECTED_TIMES[0], EXPECTED_TIMES[24]]
        session_start_returns = result.loc[
            result["Time"].isin(first_bar_times),
            "log_return",
        ]

        np.testing.assert_allclose(
            session_start_returns,
            np.log(1.001),
            rtol=0,
            atol=1e-12,
        )

    def test_filters_inactive_and_incomplete_days_and_writes_clean_csv(
        self,
    ) -> None:
        active = make_day("2024-01-02")
        inactive = make_day("2024-01-03", flat_block=2)
        incomplete = make_day("2024-01-04").drop(index=10)
        raw = pd.concat([active, inactive, incomplete], ignore_index=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            output_path = Path(temp_dir) / "processed" / "returns.csv"
            raw.to_csv(raw_path, index=False)
            result = clean_quote_file(raw_path, output_path)
            written = pd.read_csv(output_path)

        self.assertEqual(len(result), 48)
        self.assertNotIn("Unnamed: 0", written.columns)
        dates = result["Date"].dt.strftime("%Y-%m-%d").unique().tolist()
        self.assertEqual(dates, ["2024-01-02"])


if __name__ == "__main__":
    unittest.main()
