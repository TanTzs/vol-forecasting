"""Train rolling OLS/HAR candidates and save out-of-sample forecasts."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from volatility_forecasting_dqn.forecasting import (
    HARForecaster,
    OLSForecaster,
)


DATA_START = pd.Timestamp("2016-01-01")
DATA_END = pd.Timestamp("2024-12-31")
TRAIN_YEARS = 4
ROLL_MONTHS = 6
OLS_LAGS = 22
HAR_WINDOWS = {
    "1D": (1, 5, 22),  # day, week, month
    "1H": (1, 4, 20),  # hour, trading day, trading week
}


def build_rolling_windows() -> list[dict[str, pd.Timestamp]]:
    """Create four-year training windows followed by six-month forecasts."""

    training_windows = []
    train_start = DATA_START
    last_train_end = DATA_END - pd.DateOffset(months=ROLL_MONTHS)

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS) - pd.Timedelta(
            days=1
        )
        if train_end > last_train_end:
            break
        training_windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
            }
        )
        train_start += pd.DateOffset(months=ROLL_MONTHS)

    for index, window in enumerate(training_windows):
        window["forecast_start"] = window["train_end"] + pd.Timedelta(days=1)
        window["forecast_end"] = (
            training_windows[index + 1]["train_end"]
            if index + 1 < len(training_windows)
            else DATA_END
        )
    return training_windows


def load_return_files(returns_dir: Path) -> dict[str, pd.DataFrame]:
    """Load one processed return DataFrame per stock."""

    files = sorted(returns_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No return CSV files found in {returns_dir}")

    stocks = {}
    for path in files:
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"])
        stocks[path.stem] = frame
    return stocks


def save_checkpoint(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def make_forecast_records(
    stocks: dict[str, pd.DataFrame],
    ols: OLSForecaster,
    har: HARForecaster,
    window: dict[str, pd.Timestamp],
) -> list[pd.DataFrame]:
    """Generate one-step-ahead forecasts using only information available then."""

    records = []
    for stock, full_returns in stocks.items():
        available_returns = full_returns.loc[
            full_returns["Date"].le(window["forecast_end"])
        ]
        if available_returns.empty:
            continue

        ols_predictions = ols.predict(available_returns).rename(
            columns={"predicted_log_rv": "ols_prediction"}
        )
        har_predictions = har.predict(available_returns)[
            ["Date", "Time", "predicted_log_rv"]
        ].rename(columns={"predicted_log_rv": "har_prediction"})

        predictions = ols_predictions.merge(
            har_predictions,
            on=["Date", "Time"],
            how="inner",
            validate="one_to_one",
        )
        predictions = predictions.loc[
            predictions["Date"].between(
                window["forecast_start"],
                window["forecast_end"],
            )
        ].copy()
        if predictions.empty:
            continue

        predictions.insert(0, "stock", stock)
        predictions["checkpoint_date"] = window["train_end"]
        records.append(predictions)
    return records


def calculate_metrics(
    predictions: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    rows = []
    for model in ["ols", "har"]:
        forecast = predictions[f"{model}_prediction"].to_numpy()
        actual = predictions["log_rv"].to_numpy()
        difference = actual - forecast
        rows.append(
            {
                "frequency": frequency,
                "model": model,
                "n_predictions": len(predictions),
                "mse_log_rv": float(np.mean(difference**2)),
                "qlike": float(
                    np.mean(np.exp(difference) - difference - 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def train_linear_candidates(
    frequency: str,
    returns_dir: Path,
    checkpoint_root: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete rolling OLS/HAR experiment."""

    stocks = load_return_files(returns_dir)
    all_records = []
    windows = build_rolling_windows()

    for window in windows:
        train_frames = []
        for frame in stocks.values():
            training_data = frame.loc[
                frame["Date"].between(
                    window["train_start"],
                    window["train_end"],
                )
            ]
            if not training_data.empty:
                train_frames.append(training_data)

        checkpoint_name = window["train_end"].strftime("%Y%m%d")
        print(
            f"[{frequency}] training {checkpoint_name}: "
            f"{window['train_start'].date()} to {window['train_end'].date()}"
        )

        ols = OLSForecaster(
            n_lags=OLS_LAGS,
            rv_frequency=frequency,
        ).fit(train_frames)
        har = HARForecaster(
            windows=HAR_WINDOWS[frequency],
            rv_frequency=frequency,
        ).fit(train_frames)

        checkpoint_dir = checkpoint_root / frequency
        save_checkpoint(ols, checkpoint_dir / f"ols_{checkpoint_name}.pkl")
        save_checkpoint(har, checkpoint_dir / f"har_{checkpoint_name}.pkl")
        all_records.extend(
            make_forecast_records(stocks, ols, har, window)
        )

    predictions = pd.concat(all_records, ignore_index=True)
    predictions = predictions.sort_values(
        ["stock", "Date", "Time"]
    ).reset_index(drop=True)
    metrics = calculate_metrics(predictions, frequency)

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        results_dir / f"linear_predictions_{frequency}.csv",
        index=False,
    )
    metrics.to_csv(
        results_dir / f"linear_metrics_{frequency}.csv",
        index=False,
    )
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train rolling OLS and HAR volatility forecasts."
    )
    parser.add_argument(
        "--frequency",
        required=True,
        choices=["1D", "1H"],
    )
    parser.add_argument(
        "--returns-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chinese_return",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "candidates",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    args = parser.parse_args()

    predictions, metrics = train_linear_candidates(
        args.frequency,
        args.returns_dir,
        args.checkpoint_dir,
        args.results_dir,
    )
    print(f"Saved {len(predictions):,} out-of-sample predictions")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
