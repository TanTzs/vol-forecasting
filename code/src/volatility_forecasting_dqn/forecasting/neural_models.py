"""LSTM and TCN forecasters for log-realized volatility."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from volatility_forecasting_dqn.features import (
    calculate_realized_volatility,
)


def _as_return_frames(
    returns: pd.DataFrame | list[pd.DataFrame],
) -> list[pd.DataFrame]:
    if isinstance(returns, pd.DataFrame):
        return [returns]
    frames = list(returns)
    if not frames or not all(isinstance(frame, pd.DataFrame) for frame in frames):
        raise ValueError("returns must be a DataFrame or a non-empty list")
    return frames


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    return torch.device(device)


def _sliding_samples(
    values: np.ndarray,
    n_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= n_lags:
        return (
            np.empty((0, n_lags), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    windows = np.lib.stride_tricks.sliding_window_view(values, n_lags + 1)
    return (
        windows[:, :-1].astype(np.float32, copy=True),
        windows[:, -1].astype(np.float32, copy=True),
    )


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.output(sequence[:, -1])


class CausalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def _remove_future_padding(self, values: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            return values
        return values[..., :-self.padding]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.activation(
            self._remove_future_padding(self.conv1(inputs))
        )
        values = self.activation(
            self._remove_future_padding(self.conv2(values))
        )
        return self.activation(values + self.residual(inputs))


class TCNNetwork(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        num_layers: int,
    ):
        super().__init__()
        layers = []
        in_channels = 1
        for layer_index in range(num_layers):
            layers.append(
                CausalBlock(
                    in_channels=in_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                )
            )
            in_channels = num_channels
        self.tcn = nn.Sequential(*layers)
        self.output = nn.Linear(num_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.tcn(inputs)
        return self.output(sequence[:, :, -1])


class NeuralForecaster:
    """Shared pooled training and inference logic for neural forecasters."""

    model_type = "neural"

    def __init__(
        self,
        n_lags: int,
        rv_frequency: str = "1D",
        learning_rate: float = 0.001,
        epochs: int = 200,
        patience: int = 20,
        batch_size: int = 64,
        validation_start: str | pd.Timestamp | None = None,
        device: str = "auto",
        seed: int = 42,
    ):
        if n_lags <= 0:
            raise ValueError("n_lags must be positive")
        if epochs <= 0 or patience <= 0 or batch_size <= 0:
            raise ValueError("epochs, patience and batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.n_lags = n_lags
        self.rv_frequency = rv_frequency
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_start = (
            pd.Timestamp(validation_start).normalize()
            if validation_start is not None
            else None
        )
        self.device = _resolve_device(device)
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.network: nn.Module
        self.epochs_trained_: int = 0
        self.best_validation_loss_: float | None = None
        self.is_fitted_: bool = False

    def _prepare_input(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_config(self) -> dict[str, object]:
        return {
            "n_lags": self.n_lags,
            "rv_frequency": self.rv_frequency,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "validation_start": self.validation_start,
            "seed": self.seed,
        }

    def _split_frame(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rv = calculate_realized_volatility(frame, self.rv_frequency)
        values = rv["log_rv"].to_numpy(dtype=np.float32)
        features, targets = _sliding_samples(values, self.n_lags)
        if not len(features):
            empty_features = np.empty((0, self.n_lags), dtype=np.float32)
            empty_targets = np.empty(0, dtype=np.float32)
            return (
                empty_features,
                empty_targets,
                empty_features.copy(),
                empty_targets.copy(),
            )

        if self.validation_start is None:
            return (
                features,
                targets,
                np.empty((0, self.n_lags), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        target_dates = rv["Date"].iloc[self.n_lags :].to_numpy(
            dtype="datetime64[ns]"
        )
        training_mask = target_dates < self.validation_start.to_datetime64()
        return (
            features[training_mask],
            targets[training_mask],
            features[~training_mask],
            targets[~training_mask],
        )

    def _make_loader(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets).reshape(-1, 1),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def _mean_loss(self, loader: DataLoader) -> float:
        total_loss = 0.0
        total_observations = 0
        self.network.eval()
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                predictions = self.network(self._prepare_input(features))
                total_loss += nn.functional.mse_loss(
                    predictions,
                    targets,
                    reduction="sum",
                ).item()
                total_observations += len(features)
        if total_observations == 0:
            raise ValueError("Cannot compute loss on an empty DataLoader")
        return total_loss / total_observations

    def fit(
        self,
        returns: pd.DataFrame | list[pd.DataFrame],
    ) -> "NeuralForecaster":
        training_features = []
        training_targets = []
        validation_features = []
        validation_targets = []

        for frame in _as_return_frames(returns):
            train_x, train_y, val_x, val_y = self._split_frame(frame)
            if len(train_x):
                training_features.append(train_x)
                training_targets.append(train_y)
            if len(val_x):
                validation_features.append(val_x)
                validation_targets.append(val_y)

        if not training_features:
            raise ValueError("Not enough RV observations to train the model")
        if self.validation_start is not None and not validation_features:
            raise ValueError("No validation observations on or after validation_start")

        train_loader = self._make_loader(
            np.vstack(training_features),
            np.concatenate(training_targets),
            shuffle=True,
        )
        validation_loader = None
        if validation_features:
            validation_loader = self._make_loader(
                np.vstack(validation_features),
                np.concatenate(validation_targets),
                shuffle=False,
            )

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.network.to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.MSELoss()
        best_weights = deepcopy(self.network.state_dict())
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                predictions = self.network(self._prepare_input(features))
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

            self.epochs_trained_ = epoch
            if validation_loader is None:
                continue

            validation_loss = self._mean_loss(validation_loader)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = deepcopy(self.network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if validation_loader is not None:
            self.network.load_state_dict(best_weights)
            self.best_validation_loss_ = best_loss

        self.network.eval()
        self.is_fitted_ = True
        return self

    def _predict_features(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("The forecaster must be fitted before prediction")
        if features.ndim != 2 or features.shape[1] != self.n_lags:
            raise ValueError(
                f"features must have shape (n_samples, {self.n_lags})"
            )

        loader = DataLoader(
            torch.from_numpy(features.astype(np.float32, copy=False)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        predictions = []
        self.network.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                values = self.network(self._prepare_input(batch))
                predictions.append(values.squeeze(1).cpu().numpy())
        return np.concatenate(predictions)

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        rv = calculate_realized_volatility(returns, self.rv_frequency)
        values = rv["log_rv"].to_numpy(dtype=np.float32)
        features, _ = _sliding_samples(values, self.n_lags)
        if not len(features):
            raise ValueError("Not enough RV observations to predict")

        result = rv.iloc[self.n_lags :].copy()
        result["predicted_log_rv"] = self._predict_features(features)
        return result.reset_index(drop=True)

    def predict_next(self, returns: pd.DataFrame) -> float:
        values = calculate_realized_volatility(
            returns,
            self.rv_frequency,
        )["log_rv"].to_numpy(dtype=np.float32)
        return self.predict_next_from_log_rv(values)

    def predict_next_from_log_rv(
        self,
        log_rv_history: np.ndarray | list[float],
    ) -> float:
        values = np.asarray(log_rv_history, dtype=np.float32)
        if len(values) < self.n_lags:
            raise ValueError("Not enough RV observations to predict")
        features = values[-self.n_lags :].reshape(1, -1)
        return float(self._predict_features(features)[0])


class LSTMForecaster(NeuralForecaster):
    model_type = "lstm"

    def __init__(
        self,
        n_lags: int = 22,
        hidden_dim: int = 32,
        num_layers: int = 2,
        **kwargs: object,
    ):
        super().__init__(n_lags=n_lags, **kwargs)
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.network = LSTMNetwork(hidden_dim, num_layers).to(self.device)

    def _prepare_input(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unsqueeze(-1)

    def _get_config(self) -> dict[str, object]:
        return {
            **super()._get_config(),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


class TCNForecaster(NeuralForecaster):
    model_type = "tcn"

    def __init__(
        self,
        n_lags: int = 22,
        num_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 3,
        **kwargs: object,
    ):
        super().__init__(n_lags=n_lags, **kwargs)
        if num_channels <= 0 or kernel_size <= 0 or num_layers <= 0:
            raise ValueError(
                "num_channels, kernel_size and num_layers must be positive"
            )
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.network = TCNNetwork(
            num_channels,
            kernel_size,
            num_layers,
        ).to(self.device)

    def _prepare_input(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unsqueeze(1)

    def _get_config(self) -> dict[str, object]:
        return {
            **super()._get_config(),
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
        }


def save_neural_forecaster(
    forecaster: NeuralForecaster,
    path: Path,
    candidate_name: str,
) -> None:
    """Save architecture settings and CPU weights in a portable checkpoint."""

    if not forecaster.is_fitted_:
        raise ValueError("Cannot save an unfitted forecaster")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": forecaster.model_type,
            "candidate_name": candidate_name,
            "config": forecaster._get_config(),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in forecaster.network.state_dict().items()
            },
            "epochs_trained": forecaster.epochs_trained_,
            "best_validation_loss": forecaster.best_validation_loss_,
        },
        path,
    )


def load_neural_forecaster(
    path: Path,
    device: str = "auto",
) -> NeuralForecaster:
    """Restore a checkpoint created by :func:`save_neural_forecaster`."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = {
        "lstm": LSTMForecaster,
        "tcn": TCNForecaster,
    }
    try:
        forecaster_class = classes[checkpoint["model_type"]]
    except KeyError as error:
        raise ValueError("Unknown neural checkpoint model type") from error

    config = dict(checkpoint["config"])
    config["device"] = device
    forecaster = forecaster_class(**config)
    forecaster.network.load_state_dict(checkpoint["state_dict"])
    forecaster.network.eval()
    forecaster.epochs_trained_ = int(checkpoint["epochs_trained"])
    forecaster.best_validation_loss_ = checkpoint["best_validation_loss"]
    forecaster.is_fitted_ = True
    return forecaster
