"""Prepare aligned candidate forecasts for one-step DQN experiments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KEY_COLUMNS = ["stock", "Date", "Time", "checkpoint_date"]
ACTUAL_COLUMNS = ["rv", "log_rv"]
LINEAR_PREDICTIONS = ["ols_prediction", "har_prediction"]
NEURAL_PREDICTIONS = [
    "lstm_prediction",
    "tcn_short_prediction",
    "tcn_medium_prediction",
    "tcn_long_prediction",
]
PREDICTION_COLUMNS = LINEAR_PREDICTIONS + NEURAL_PREDICTIONS


def load_predictions(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a prediction file and check its required columns and keys."""

    frame = pd.read_csv(path, dtype={"stock": str})
    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{path.name} contains missing merge keys")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path.name} contains duplicate merge keys")
    return frame


def prepare_one_step_dqn_data(
    frequency: str,
    results_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Combine the six one-step forecasts for one sampling frequency."""

    linear = load_predictions(
        results_dir / f"linear_predictions_{frequency}.csv",
        KEY_COLUMNS + ACTUAL_COLUMNS + LINEAR_PREDICTIONS,
    )
    neural = load_predictions(
        results_dir / f"neural_predictions_{frequency}.csv",
        KEY_COLUMNS + ACTUAL_COLUMNS + NEURAL_PREDICTIONS,
    )

    merged = linear.merge(
        neural,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_linear", "_neural"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        raise ValueError(
            f"{unmatched.sum():,} prediction rows are not shared by both files"
        )

    for column in ACTUAL_COLUMNS:
        linear_values = merged[f"{column}_linear"].to_numpy()
        neural_values = merged[f"{column}_neural"].to_numpy()
        if not np.allclose(linear_values, neural_values, rtol=0, atol=1e-12):
            raise ValueError(f"{column} differs between linear and neural files")
        merged[column] = linear_values

    output_columns = (
        KEY_COLUMNS + ACTUAL_COLUMNS + PREDICTION_COLUMNS
    )
    one_step_data = merged[output_columns].copy()
    if one_step_data[ACTUAL_COLUMNS + PREDICTION_COLUMNS].isna().any().any():
        raise ValueError("The merged data contain missing targets or predictions")

    dates = pd.to_datetime(one_step_data["Date"])
    checkpoint_dates = pd.to_datetime(one_step_data["checkpoint_date"])
    if checkpoint_dates.ge(dates).any():
        raise ValueError("A forecast is dated on or before its training checkpoint")

    one_step_data = one_step_data.sort_values(
        ["stock", "Date", "Time"]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"one_step_dqn_data_{frequency}.csv"
    one_step_data.to_csv(output_path, index=False)
    print(f"Saved {len(one_step_data):,} aligned rows to {output_path}")
    return one_step_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare candidate forecasts for one-step DQN training."
    )
    parser.add_argument(
        "--frequency",
        required=True,
        choices=["1D", "1H"],
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    args = parser.parse_args()

    prepare_one_step_dqn_data(
        frequency=args.frequency,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
