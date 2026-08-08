import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts"))

from train_one_step_dqn import build_parser, train  # noqa: E402


PREDICTION_COLUMNS = [
    "ols_prediction",
    "har_prediction",
    "lstm_prediction",
    "tcn_short_prediction",
    "tcn_medium_prediction",
    "tcn_long_prediction",
]


def make_one_step_data() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2023-10-02", "2024-01-05")
    for step, date in enumerate(dates):
        actual = float(np.sin(step / 10))
        row = {
            "stock": "000001",
            "Date": date,
            "Time": "15:00:00",
            "checkpoint_date": "2023-06-30",
            "rv": float(np.exp(actual)),
            "log_rv": actual,
        }
        for model, column in enumerate(PREDICTION_COLUMNS):
            row[column] = actual + 0.1 * model
        rows.append(row)
    return pd.DataFrame(rows)


class OneStepTrainingScriptTests(unittest.TestCase):
    def test_validation_defaults_match_training_schedule(self) -> None:
        args = build_parser().parse_args(["--frequency", "1D"])

        self.assertEqual(args.eval_interval, 1_000)
        self.assertEqual(args.patience, 50)
        self.assertEqual(args.train_frequency, 4)

    def test_short_training_run_saves_history_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            data_dir = root / "data"
            checkpoint_dir = root / "checkpoints"
            results_dir = root / "results"
            data_dir.mkdir()
            make_one_step_data().to_csv(
                data_dir / "one_step_dqn_data_1D.csv",
                index=False,
            )

            args = build_parser().parse_args(
                [
                    "--frequency",
                    "1D",
                    "--training-steps",
                    "4",
                    "--batch-size",
                    "2",
                    "--replay-capacity",
                    "4",
                    "--learning-starts",
                    "2",
                    "--target-update-interval",
                    "2",
                    "--epsilon-decay-steps",
                    "4",
                    "--eval-interval",
                    "4",
                    "--patience",
                    "0",
                    "--device",
                    "cpu",
                    "--no-deterministic",
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--results-dir",
                    str(results_dir),
                ]
            )

            history, checkpoint_path = train(args)

            self.assertEqual(len(history), 1)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(
                (
                    results_dir
                    / "dqn_training_one_step_1D_qlike_seed42.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    results_dir
                    / "dqn_config_one_step_1D_qlike_seed42.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
