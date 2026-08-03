"""Small numerical and serialization checks for neural forecasters."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts"))

from volatility_forecasting_dqn.data.returns import EXPECTED_TIMES  # noqa: E402
from volatility_forecasting_dqn.forecasting import (  # noqa: E402
    LSTMForecaster,
    TCNForecaster,
    load_neural_forecaster,
    save_neural_forecaster,
)
from train_neural_candidates import create_forecaster  # noqa: E402


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


class NeuralForecasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log_rv = (
            -6.0
            + 0.2 * np.sin(np.arange(30) / 3)
            + 0.01 * np.arange(30)
        ).tolist()
        self.returns = returns_from_log_rv(self.log_rv)
        self.validation_start = pd.bdate_range(
            "2020-01-02",
            periods=21,
        )[-1]

    def _common_arguments(self) -> dict[str, object]:
        return {
            "n_lags": 3,
            "epochs": 2,
            "patience": 1,
            "batch_size": 8,
            "validation_start": self.validation_start,
            "device": "cpu",
            "seed": 7,
        }

    def test_lstm_fit_predict_and_checkpoint_round_trip(self) -> None:
        model = LSTMForecaster(
            hidden_dim=4,
            num_layers=1,
            **self._common_arguments(),
        ).fit(self.returns)

        predictions = model.predict(self.returns)
        next_prediction = model.predict_next_from_log_rv(self.log_rv)

        self.assertEqual(len(predictions), len(self.log_rv) - 3)
        self.assertTrue(np.isfinite(predictions["predicted_log_rv"]).all())
        self.assertTrue(np.isfinite(next_prediction))
        self.assertGreater(model.epochs_trained_, 0)
        self.assertIsNotNone(model.best_validation_loss_)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lstm.pt"
            save_neural_forecaster(model, path, "lstm")
            restored = load_neural_forecaster(path, device="cpu")

            self.assertAlmostEqual(
                restored.predict_next_from_log_rv(self.log_rv),
                next_prediction,
                places=6,
            )

    def test_tcn_fit_and_predict(self) -> None:
        model = TCNForecaster(
            num_channels=4,
            kernel_size=2,
            num_layers=1,
            **self._common_arguments(),
        ).fit(self.returns)

        predictions = model.predict(self.returns)
        self.assertEqual(len(predictions), len(self.log_rv) - 3)
        self.assertTrue(np.isfinite(predictions["predicted_log_rv"]).all())

    def test_candidate_architectures_match_the_paper(self) -> None:
        expected = {
            "lstm": (LSTMForecaster, 22),
            "tcn_short": (TCNForecaster, 5),
            "tcn_medium": (TCNForecaster, 22),
            "tcn_long": (TCNForecaster, 60),
        }
        for candidate_name, (expected_class, expected_lags) in expected.items():
            model = create_forecaster(
                candidate_name=candidate_name,
                frequency="1D",
                validation_start=pd.Timestamp("2019-07-01"),
                learning_rate=0.001,
                epochs=1,
                patience=1,
                batch_size=8,
                device="cpu",
                seed=42,
            )
            self.assertIsInstance(model, expected_class)
            self.assertEqual(model.n_lags, expected_lags)


if __name__ == "__main__":
    unittest.main()
