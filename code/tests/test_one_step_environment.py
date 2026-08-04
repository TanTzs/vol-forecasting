import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from volatility_forecasting_dqn.rl import (  # noqa: E402
    OneStepEnvironment,
    OneStepStateDataset,
)


PREDICTION_COLUMNS = [
    "ols_prediction",
    "har_prediction",
    "lstm_prediction",
    "tcn_short_prediction",
    "tcn_medium_prediction",
    "tcn_long_prediction",
]


def make_panel() -> pd.DataFrame:
    dates = [
        "2023-12-28",
        "2023-12-29",
        "2023-12-31",
        "2024-01-02",
        "2024-07-01",
    ]
    rows = []
    for step, date in enumerate(dates):
        actual = float(step)
        row = {
            "stock": "000001",
            "Date": date,
            "Time": "15:00:00",
            "checkpoint_date": "2023-06-30",
            "rv": float(np.exp(actual)),
            "log_rv": actual,
        }
        for model, column in enumerate(PREDICTION_COLUMNS):
            row[column] = actual + model
        rows.append(row)
    return pd.DataFrame(rows)


class OneStepEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset = OneStepStateDataset(make_panel(), lookback=2)
        self.dataset = dataset

    def test_zero_loss_action_has_zero_reward(self) -> None:
        environment = OneStepEnvironment(self.dataset, mode="train")
        state, reset_info = environment.reset(seed=42)

        self.assertEqual(state.shape, (22,))
        self.assertEqual(environment.observation_space.shape, (22,))
        self.assertEqual(environment.action_space.n, 6)
        self.assertNotIn("actual_log_rv", reset_info)

        terminal_state, reward, terminated, truncated, info = (
            environment.step(0)
        )

        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_array_equal(terminal_state, np.zeros(22))
        self.assertEqual(info["selected_model"], "ols")
        self.assertEqual(info["selected_qlike"], 0.0)

    def test_qlike_reward_is_negative_selected_qlike(self) -> None:
        environment = OneStepEnvironment(
            self.dataset,
            mode="validation",
            reward_metric="qlike",
        )
        environment.reset()
        _, reward, _, _, info = environment.step(1)

        expected_qlike = np.exp(-1.0)
        self.assertAlmostEqual(info["selected_qlike"], expected_qlike, places=6)
        self.assertAlmostEqual(reward, -expected_qlike, places=6)
        self.assertEqual(info["reward_metric"], "qlike")

    def test_mse_reward_is_negative_selected_mse(self) -> None:
        environment = OneStepEnvironment(
            self.dataset,
            mode="validation",
            reward_metric="mse",
        )
        environment.reset()
        _, reward, _, _, info = environment.step(1)

        self.assertEqual(info["selected_mse"], 1.0)
        self.assertEqual(info["selected_loss"], 1.0)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["reward_metric"], "mse")

    def test_validation_and_test_use_disjoint_dates(self) -> None:
        validation = OneStepEnvironment(self.dataset, mode="validation")
        test = OneStepEnvironment(self.dataset, mode="test")

        _, validation_info = validation.reset()
        _, test_info = test.reset()

        self.assertLessEqual(
            validation_info["Date"],
            pd.Timestamp("2024-06-30"),
        )
        self.assertGreaterEqual(
            test_info["Date"],
            pd.Timestamp("2024-07-01"),
        )

    def test_gymnasium_contract(self) -> None:
        environment = OneStepEnvironment(self.dataset, mode="train")
        check_env(environment, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
