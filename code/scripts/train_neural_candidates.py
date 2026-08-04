"""Train rolling LSTM/TCN candidates and save out-of-sample forecasts."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from volatility_forecasting_dqn.forecasting import (  # noqa: E402
    LSTMForecaster,
    TCNForecaster,
    save_neural_forecaster,
)
from train_linear_candidates import (  # noqa: E402
    build_rolling_windows,
    load_return_files,
)


VALIDATION_MONTHS = 6
CANDIDATE_NAMES = (
    "lstm",
    "tcn_short",
    "tcn_medium",
    "tcn_long",
)


def create_forecaster(
    candidate_name: str,
    frequency: str,
    validation_start: pd.Timestamp,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    seed: int,
) -> LSTMForecaster | TCNForecaster:
    """Build one of the four neural candidates used by the paper."""

    common = {
        "rv_frequency": frequency,
        "validation_start": validation_start,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
    }
    if candidate_name == "lstm":
        return LSTMForecaster(
            n_lags=22,
            hidden_dim=32,
            num_layers=2,
            **common,
        )

    tcn_settings = {
        "tcn_short": {"n_lags": 5, "num_layers": 1},
        "tcn_medium": {"n_lags": 22, "num_layers": 3},
        "tcn_long": {"n_lags": 60, "num_layers": 4},
    }
    if candidate_name not in tcn_settings:
        raise ValueError(f"Unknown candidate: {candidate_name}")
    return TCNForecaster(
        num_channels=32,
        kernel_size=3,
        **tcn_settings[candidate_name],
        **common,
    )


def make_forecast_records(
    stocks: dict[str, pd.DataFrame],
    models: dict[str, LSTMForecaster | TCNForecaster],
    window: dict[str, pd.Timestamp],
) -> list[pd.DataFrame]:
    """Generate walk-forward one-step predictions for the next six months."""

    records = []
    for stock, full_returns in stocks.items():
        available_returns = full_returns.loc[
            full_returns["Date"].le(window["forecast_end"])
        ]
        if available_returns.empty:
            continue

        predictions = None
        for candidate_name, model in models.items():
            current = model.predict(available_returns)
            current = current.rename(
                columns={
                    "predicted_log_rv": f"{candidate_name}_prediction",
                }
            )
            if predictions is None:
                predictions = current
            else:
                predictions = predictions.merge(
                    current[
                        [
                            "Date",
                            "Time",
                            f"{candidate_name}_prediction",
                        ]
                    ],
                    on=["Date", "Time"],
                    how="inner",
                    validate="one_to_one",
                )

        if predictions is None:
            continue
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
    candidate_names: tuple[str, ...] = CANDIDATE_NAMES,
) -> pd.DataFrame:
    rows = []
    actual = predictions["log_rv"].to_numpy()
    for candidate_name in candidate_names:
        difference = (
            actual
            - predictions[f"{candidate_name}_prediction"].to_numpy()
        )
        rows.append(
            {
                "frequency": frequency,
                "model": candidate_name,
                "n_predictions": len(predictions),
                "mse_log_rv": float(np.mean(difference**2)),
                "qlike": float(
                    np.mean(np.exp(difference) - difference - 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def train_neural_candidates(
    frequency: str,
    returns_dir: Path,
    checkpoint_root: Path,
    results_dir: Path,
    learning_rate: float = 0.001,
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 42,
    candidate_names: tuple[str, ...] = CANDIDATE_NAMES,
    checkpoint_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pooled rolling experiment for selected neural candidates."""

    stocks = load_return_files(returns_dir)
    windows = build_rolling_windows()
    if checkpoint_date is not None:
        checkpoint_date = pd.Timestamp(checkpoint_date).normalize()
        windows = [
            window
            for window in windows
            if window["train_end"] == checkpoint_date
        ]
        if not windows:
            raise ValueError(
                f"No rolling checkpoint ends on {checkpoint_date.date()}"
            )

    unknown_candidates = set(candidate_names) - set(CANDIDATE_NAMES)
    if unknown_candidates:
        raise ValueError(
            f"Unknown candidates: {sorted(unknown_candidates)}"
        )
    if not candidate_names:
        raise ValueError("At least one candidate must be selected")

    is_full_run = (
        candidate_names == CANDIDATE_NAMES
        and checkpoint_date is None
    )
    if is_full_run:
        run_tag = frequency
        checkpoint_output_root = checkpoint_root / frequency
    else:
        tag_parts = [
            frequency,
            *candidate_names,
            (
                checkpoint_date.strftime("%Y%m%d")
                if checkpoint_date is not None
                else "all-checkpoints"
            ),
            f"bs{batch_size}",
            f"ep{epochs}",
            f"seed{seed}",
        ]
        run_tag = "_".join(tag_parts)
        checkpoint_output_root = checkpoint_root / frequency / run_tag

    all_records = []

    for window in windows:
        training_frames = []
        for frame in stocks.values():
            training_data = frame.loc[
                frame["Date"].between(
                    window["train_start"],
                    window["train_end"],
                )
            ]
            if not training_data.empty:
                training_frames.append(training_data)

        validation_start = (
            window["train_end"]
            - pd.DateOffset(months=VALIDATION_MONTHS)
            + pd.Timedelta(days=1)
        )
        checkpoint_name = window["train_end"].strftime("%Y%m%d")
        print(
            f"[{frequency}] checkpoint {checkpoint_name}: "
            f"train from {window['train_start'].date()}, "
            f"validate from {validation_start.date()}"
        )

        models = {}
        for candidate_name in candidate_names:
            print(f"  training {candidate_name} ...")
            model = create_forecaster(
                candidate_name=candidate_name,
                frequency=frequency,
                validation_start=validation_start,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                device=device,
                seed=seed,
            ).fit(training_frames)
            models[candidate_name] = model

            checkpoint_path = (
                checkpoint_output_root
                / f"{candidate_name}_{checkpoint_name}.pt"
            )
            save_neural_forecaster(
                model,
                checkpoint_path,
                candidate_name,
            )
            validation_loss = model.best_validation_loss_
            print(
                f"  saved {checkpoint_path.name} "
                f"(epochs={model.epochs_trained_}, "
                f"validation_mse={validation_loss:.6f})"
            )

        all_records.extend(
            make_forecast_records(stocks, models, window)
        )

    predictions = pd.concat(all_records, ignore_index=True)
    predictions = predictions.sort_values(
        ["stock", "Date", "Time"]
    ).reset_index(drop=True)
    metrics = calculate_metrics(
        predictions,
        frequency,
        candidate_names,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        results_dir / f"neural_predictions_{run_tag}.csv",
        index=False,
    )
    metrics.to_csv(
        results_dir / f"neural_metrics_{run_tag}.csv",
        index=False,
    )
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train rolling pooled LSTM and TCN candidates."
    )
    parser.add_argument(
        "--frequency",
        required=True,
        choices=["1D", "1H"],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate",
        choices=CANDIDATE_NAMES,
        help="Train only one candidate; omit to train all candidates.",
    )
    parser.add_argument(
        "--checkpoint-date",
        type=pd.Timestamp,
        help="Train only the rolling checkpoint ending on YYYY-MM-DD.",
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

    predictions, metrics = train_neural_candidates(
        frequency=args.frequency,
        returns_dir=args.returns_dir,
        checkpoint_root=args.checkpoint_dir,
        results_dir=args.results_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        candidate_names=(
            (args.candidate,)
            if args.candidate is not None
            else CANDIDATE_NAMES
        ),
        checkpoint_date=args.checkpoint_date,
    )
    print(f"Saved {len(predictions):,} out-of-sample predictions")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
