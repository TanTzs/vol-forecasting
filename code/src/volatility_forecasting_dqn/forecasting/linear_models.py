"""OLS and HAR forecasting baselines for log-realized volatility."""

import numpy as np
import pandas as pd

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


def _fit_linear_regression(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(features)), features])
    parameters = np.linalg.lstsq(design, targets, rcond=None)[0]
    return float(parameters[0]), parameters[1:]


def _linear_prediction(
    features: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    return intercept + features @ coefficients


class OLSForecaster:
    """Forecast log-RV from its most recent ``n_lags`` observations."""

    def __init__(self, n_lags: int = 22, rv_frequency: str = "1D"):
        if n_lags <= 0:
            raise ValueError("n_lags must be positive")
        self.n_lags = n_lags
        self.rv_frequency = rv_frequency
        self.intercept_: float | None = None
        self.coefficients_: np.ndarray | None = None

    def fit(
        self,
        returns: pd.DataFrame | list[pd.DataFrame],
    ) -> "OLSForecaster":
        features = []
        targets = []

        for frame in _as_return_frames(returns):
            values = calculate_realized_volatility(
                frame,
                self.rv_frequency,
            )["log_rv"].to_numpy()
            if len(values) <= self.n_lags:
                continue
            features.append(
                np.array(
                    [
                        values[index - self.n_lags : index]
                        for index in range(self.n_lags, len(values))
                    ]
                )
            )
            targets.append(values[self.n_lags :])

        if not features:
            raise ValueError("Not enough RV observations to fit OLS")

        self.intercept_, self.coefficients_ = _fit_linear_regression(
            np.vstack(features),
            np.concatenate(targets),
        )
        return self

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        rv = calculate_realized_volatility(returns, self.rv_frequency)
        values = rv["log_rv"].to_numpy()
        if len(values) <= self.n_lags:
            raise ValueError("Not enough RV observations to predict")

        features = np.array(
            [
                values[index - self.n_lags : index]
                for index in range(self.n_lags, len(values))
            ]
        )
        result = rv.iloc[self.n_lags :].copy()
        result["predicted_log_rv"] = self._predict_features(features)
        return result.reset_index(drop=True)

    def predict_next(self, returns: pd.DataFrame) -> float:
        values = calculate_realized_volatility(
            returns,
            self.rv_frequency,
        )["log_rv"].to_numpy()
        if len(values) < self.n_lags:
            raise ValueError("Not enough RV observations to predict")
        features = values[-self.n_lags :].reshape(1, -1)
        return float(self._predict_features(features)[0])

    def _predict_features(self, features: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coefficients_ is None:
            raise ValueError("The forecaster must be fitted before prediction")
        return _linear_prediction(
            features,
            self.intercept_,
            self.coefficients_,
        )


class HARForecaster:
    """HAR forecast using short-, medium- and long-window log-RV means."""

    def __init__(
        self,
        windows: tuple[int, ...] = (1, 5, 22),
        rv_frequency: str = "1D",
    ):
        if not windows or any(window <= 0 for window in windows):
            raise ValueError("HAR windows must contain positive integers")
        self.windows = tuple(windows)
        self.max_window = max(windows)
        self.rv_frequency = rv_frequency
        self.intercept_: float | None = None
        self.coefficients_: np.ndarray | None = None

    def fit(
        self,
        returns: pd.DataFrame | list[pd.DataFrame],
    ) -> "HARForecaster":
        features = []
        targets = []

        for frame in _as_return_frames(returns):
            values = calculate_realized_volatility(
                frame,
                self.rv_frequency,
            )["log_rv"].to_numpy()
            if len(values) <= self.max_window:
                continue
            features.append(self._make_features(values))
            targets.append(values[self.max_window :])

        if not features:
            raise ValueError("Not enough RV observations to fit HAR")

        self.intercept_, self.coefficients_ = _fit_linear_regression(
            np.vstack(features),
            np.concatenate(targets),
        )
        return self

    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        rv = calculate_realized_volatility(returns, self.rv_frequency)
        values = rv["log_rv"].to_numpy()
        if len(values) <= self.max_window:
            raise ValueError("Not enough RV observations to predict")

        result = rv.iloc[self.max_window :].copy()
        result["predicted_log_rv"] = self._predict_features(
            self._make_features(values)
        )
        return result.reset_index(drop=True)

    def predict_next(self, returns: pd.DataFrame) -> float:
        values = calculate_realized_volatility(
            returns,
            self.rv_frequency,
        )["log_rv"].to_numpy()
        if len(values) < self.max_window:
            raise ValueError("Not enough RV observations to predict")
        features = np.array(
            [[values[-window:].mean() for window in self.windows]]
        )
        return float(self._predict_features(features)[0])

    def _make_features(self, values: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    values[index - window : index].mean()
                    for window in self.windows
                ]
                for index in range(self.max_window, len(values))
            ]
        )

    def _predict_features(self, features: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coefficients_ is None:
            raise ValueError("The forecaster must be fitted before prediction")
        return _linear_prediction(
            features,
            self.intercept_,
            self.coefficients_,
        )
