"""One-step environment for candidate-model selection."""

from numbers import Integral

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .one_step_state import PREDICTION_COLUMNS, OneStepStateDataset


MODEL_NAMES = tuple(
    column.removesuffix("_prediction")
    for column in PREDICTION_COLUMNS
)
REWARD_METRICS = ("mse", "qlike")
DATE_SPLITS = {
    "train": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")),
    "validation": (
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-06-30"),
    ),
    "test": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
}


class OneStepEnvironment(gym.Env):
    """Expose one-step forecast selection through ``reset`` and ``step``.

    Training resets sample target rows randomly with replacement. Validation
    and test resets traverse their target rows once in chronological panel
    order. Each episode terminates immediately after one action.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: OneStepStateDataset,
        mode: str,
        reward_metric: str = "qlike",
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in DATE_SPLITS:
            raise ValueError(
                f"mode must be one of {sorted(DATE_SPLITS)}, got {mode!r}"
            )
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be positive and finite")
        if reward_metric not in REWARD_METRICS:
            raise ValueError(
                f"reward_metric must be one of {REWARD_METRICS}, "
                f"got {reward_metric!r}"
            )

        self.dataset = dataset
        self.mode = mode
        self.reward_metric = reward_metric
        self.reward_scale = float(reward_scale)

        target_dates = dataset.data.iloc[dataset.target_rows]["Date"]
        start_date, end_date = DATE_SPLITS[mode]
        in_split = target_dates.between(start_date, end_date)
        self.sample_indices = np.flatnonzero(in_split.to_numpy())
        if len(self.sample_indices) == 0:
            raise ValueError(f"No one-step samples are available for {mode}")

        self._cursor = 0
        self._current_sample: dict[str, object] | None = None
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dataset.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(MODEL_NAMES))

    @property
    def n_actions(self) -> int:
        return len(MODEL_NAMES)

    @property
    def state_dim(self) -> int:
        return self.dataset.state_dim

    @property
    def exhausted(self) -> bool:
        return self.mode != "train" and self._cursor >= len(
            self.sample_indices
        )

    def restart(self) -> None:
        """Restart sequential validation or test traversal."""

        self._cursor = 0
        self._current_sample = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Start a one-step episode and return its observable state."""

        super().reset(seed=seed)

        if self.mode == "train":
            position = int(self.np_random.integers(len(self.sample_indices)))
        else:
            if self.exhausted:
                raise StopIteration(
                    f"{self.mode} samples are exhausted; call restart()"
                )
            position = self._cursor
            self._cursor += 1

        dataset_index = int(self.sample_indices[position])
        self._current_sample = self.dataset[dataset_index]
        state = self._current_sample["state"].copy()
        info = {
            key: self._current_sample[key]
            for key in ["stock", "Date", "Time", "checkpoint_date"]
        }
        return state, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Evaluate one selected candidate and terminate the episode."""

        if self._current_sample is None:
            raise RuntimeError("Call reset() before step()")
        if not isinstance(action, Integral) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be between 0 and {self.n_actions - 1}")

        sample = self._current_sample
        predictions = sample["predictions"]
        actual = float(sample["actual_log_rv"])
        differences = actual - predictions
        qlike = np.exp(differences) - differences - 1.0
        squared_errors = differences**2

        selected_mse = float(squared_errors[action])
        selected_qlike = float(qlike[action])
        selected_loss = (
            selected_mse
            if self.reward_metric == "mse"
            else selected_qlike
        )
        reward = -self.reward_scale * selected_loss

        info = {
            "stock": sample["stock"],
            "Date": sample["Date"],
            "Time": sample["Time"],
            "checkpoint_date": sample["checkpoint_date"],
            "action": action,
            "selected_model": MODEL_NAMES[action],
            "actual_log_rv": actual,
            "selected_prediction": float(predictions[action]),
            "selected_mse": selected_mse,
            "selected_qlike": selected_qlike,
            "reward_metric": self.reward_metric,
            "selected_loss": selected_loss,
            "all_predictions": dict(zip(MODEL_NAMES, predictions.tolist())),
            "all_mse": dict(zip(MODEL_NAMES, squared_errors.tolist())),
            "all_qlike": dict(zip(MODEL_NAMES, qlike.tolist())),
        }

        self._current_sample = None
        terminal_state = np.zeros(self.state_dim, dtype=np.float32)
        return terminal_state, float(reward), True, False, info
