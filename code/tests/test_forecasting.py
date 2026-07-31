"""Numerical checks for RV aggregation and linear forecasters."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from volatility_forecasting_dqn.data.returns import EXPECTED_TIMES  # noqa: E402
from volatility_forecasting_dqn.features import (  # noqa: E402
    calculate_realized_volatility,
)
from volatility_forecasting_dqn.forecasting import (  # noqa: E402
    HARForecaster,
    OLSForecaster,
)


def returns_from_log_rv(log_rv_values: list[float]) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2020-01-02", periods=len(log_rv_values))
    for trading_date, log_rv in zip(dates, log_rv_values):
        intraday_return = np.sqrt(np.exp(log_rv) / len(EXPECTED_TIMES))
        for bar_time in EXPECTED_TIMES:
            rows.append(
                {
                    "Date": trading_date,
                    "Time": bar_time,
                    "log_return": intraday_return,
                }
            )
    return pd.DataFrame(rows)


class RealizedVolatilityTests(unittest.TestCase):
    def test_daily_and_hourly_aggregation(self) -> None:
        intraday_returns = np.repeat([0.01, 0.02, 0.03, 0.04], 12)
        returns = pd.DataFrame(
            {
                "Date": pd.Timestamp("2024-01-02"),
                "Time": EXPECTED_TIMES,
                "log_return": intraday_returns,
            }
        )

        daily = calculate_realized_volatility(returns, "1D")
        hourly = calculate_realized_volatility(returns, "1H")

        expected_hourly_rv = 12 * np.array([0.01, 0.02, 0.03, 0.04]) ** 2
        self.assertEqual(len(daily), 1)
        self.assertEqual(len(hourly), 4)
        np.testing.assert_allclose(hourly["rv"], expected_hourly_rv)
        self.assertAlmostEqual(daily.loc[0, "rv"], expected_hourly_rv.sum())
        np.testing.assert_allclose(hourly["log_rv"], np.log(expected_hourly_rv))
        self.assertEqual(
            hourly["Time"].tolist(),
            [EXPECTED_TIMES[11], EXPECTED_TIMES[23],
             EXPECTED_TIMES[35], EXPECTED_TIMES[47]],
        )


class LinearForecasterTests(unittest.TestCase):
    def test_ols_recovers_an_exact_two_lag_process(self) -> None:
        log_rv = [-6.0, -5.8]
        for _ in range(78):
            log_rv.append(0.2 + 0.3 * log_rv[-2] + 0.6 * log_rv[-1])
        returns = returns_from_log_rv(log_rv)

        model = OLSForecaster(n_lags=2).fit(returns)
        predictions = model.predict(returns)
        expected_next = 0.2 + 0.3 * log_rv[-2] + 0.6 * log_rv[-1]

        np.testing.assert_allclose(
            predictions["predicted_log_rv"],
            log_rv[2:],
            atol=1e-10,
        )
        self.assertEqual(
            predictions.loc[0, "Date"],
            pd.bdate_range("2020-01-02", periods=3)[-1],
        )
        self.assertAlmostEqual(
            model.predict_next(returns),
            expected_next,
            places=10,
        )

    def test_har_recovers_an_exact_1_5_22_process(self) -> None:
        log_rv = np.linspace(-6.2, -5.6, 22).tolist()
        for _ in range(78):
            log_rv.append(
                0.1
                + 0.5 * log_rv[-1]
                + 0.3 * np.mean(log_rv[-5:])
                + 0.1 * np.mean(log_rv[-22:])
            )
        returns = returns_from_log_rv(log_rv)

        model = HARForecaster().fit(returns)
        predictions = model.predict(returns)
        expected_next = (
            0.1
            + 0.5 * log_rv[-1]
            + 0.3 * np.mean(log_rv[-5:])
            + 0.1 * np.mean(log_rv[-22:])
        )

        np.testing.assert_allclose(
            predictions["predicted_log_rv"],
            log_rv[22:],
            atol=1e-10,
        )
        self.assertAlmostEqual(
            model.predict_next(returns),
            expected_next,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
