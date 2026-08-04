import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from volatility_forecasting_dqn.rl import OneStepStateDataset  # noqa: E402


PREDICTION_COLUMNS = [
    "ols_prediction",
    "har_prediction",
    "lstm_prediction",
    "tcn_short_prediction",
    "tcn_medium_prediction",
    "tcn_long_prediction",
]


def make_panel() -> pd.DataFrame:
    rows = []
    for stock in ["000001", "600010"]:
        for step in range(4):
            actual = float(step + 1)
            row = {
                "stock": stock,
                "Date": f"2020-01-0{step + 2}",
                "Time": "15:00:00",
                "checkpoint_date": "2019-12-31",
                "rv": float(np.exp(actual)),
                "log_rv": actual,
            }
            for model, column in enumerate(PREDICTION_COLUMNS):
                row[column] = actual + model + 0.5
            rows.append(row)
    return pd.DataFrame(rows)


class OneStepStateDatasetTests(unittest.TestCase):
    def test_state_uses_only_the_same_stocks_past(self) -> None:
        dataset = OneStepStateDataset(make_panel(), lookback=2)

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.state_dim, 22)

        first_stock = dataset[0]
        temporal = first_stock["state"][:-6].reshape(2, 8)

        np.testing.assert_array_equal(temporal[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(temporal[:, 1], [0.5, 0.5])
        np.testing.assert_array_equal(temporal[:, -1], [1.0, 1.0])
        np.testing.assert_array_equal(
            first_stock["predictions"],
            np.arange(6) + 3.5,
        )
        self.assertEqual(first_stock["actual_log_rv"], 3.0)
        self.assertEqual(first_stock["stock"], "000001")

        second_stock = dataset[2]
        second_temporal = second_stock["state"][:-6].reshape(2, 8)
        np.testing.assert_array_equal(second_temporal[:, 0], [1.0, 2.0])
        self.assertEqual(second_stock["stock"], "600010")


if __name__ == "__main__":
    unittest.main()
