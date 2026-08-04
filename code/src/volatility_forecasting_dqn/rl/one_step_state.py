"""State construction for one-step candidate-model selection."""

import numpy as np
import pandas as pd


LOOKBACK = 60
PREDICTION_COLUMNS = [
    "ols_prediction",
    "har_prediction",
    "lstm_prediction",
    "tcn_short_prediction",
    "tcn_medium_prediction",
    "tcn_long_prediction",
]
TIME_COLUMNS = ["stock", "Date", "Time"]
REQUIRED_COLUMNS = (
    TIME_COLUMNS
    + ["checkpoint_date", "rv", "log_rv"]
    + PREDICTION_COLUMNS
)


class OneStepStateDataset:
    """Construct leakage-free one-step states from an aligned forecast panel.

    For a target row ``t``, the state contains:

    - the previous ``lookback`` log-RVs;
    - the six corresponding out-of-sample forecast errors;
    - a fidelity indicator equal to one for every realized history row;
    - the six candidate forecasts for target ``t``.

    The first ``lookback`` rows of each stock are used only as warm-up.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int = LOOKBACK,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")

        missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing one-step data columns: {sorted(missing_columns)}"
            )

        frame = data[REQUIRED_COLUMNS].copy()
        if frame[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError("One-step data contain missing values")
        if frame.duplicated(TIME_COLUMNS).any():
            raise ValueError("One-step data contain duplicate stock-time rows")

        frame["stock"] = frame["stock"].astype(str)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame = frame.sort_values(TIME_COLUMNS).reset_index(drop=True)

        numeric_columns = ["rv", "log_rv"] + PREDICTION_COLUMNS
        numeric = frame[numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("One-step data contain non-finite values")
        frame[numeric_columns] = numeric

        self.lookback = lookback
        self.data = frame
        self.log_rv = frame["log_rv"].to_numpy(dtype=np.float32)
        self.predictions = frame[PREDICTION_COLUMNS].to_numpy(
            dtype=np.float32
        )
        self.errors = self.predictions - self.log_rv[:, None]

        target_rows = []
        for positions in frame.groupby("stock", sort=False).indices.values():
            positions = np.asarray(positions)
            if len(positions) > lookback:
                target_rows.append(positions[lookback:])
        if not target_rows:
            raise ValueError("No stock has enough rows for the lookback")
        self.target_rows = np.concatenate(target_rows)

    @classmethod
    def from_csv(
        cls,
        path: str,
        lookback: int = LOOKBACK,
    ) -> "OneStepStateDataset":
        data = pd.read_csv(path, dtype={"stock": str})
        return cls(data, lookback=lookback)

    @property
    def state_dim(self) -> int:
        n_models = len(PREDICTION_COLUMNS)
        return self.lookback * (n_models + 2) + n_models

    def __len__(self) -> int:
        return len(self.target_rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        target_row = int(self.target_rows[index])
        history = slice(target_row - self.lookback, target_row)

        temporal = np.empty(
            (self.lookback, len(PREDICTION_COLUMNS) + 2),
            dtype=np.float32,
        )
        temporal[:, 0] = self.log_rv[history]
        temporal[:, 1:-1] = self.errors[history]
        temporal[:, -1] = 1.0

        state = np.concatenate(
            [temporal.reshape(-1), self.predictions[target_row]]
        ).astype(np.float32, copy=False)
        row = self.data.iloc[target_row]

        return {
            "state": state,
            "predictions": self.predictions[target_row].copy(),
            "actual_log_rv": np.float32(self.log_rv[target_row]),
            "stock": row["stock"],
            "Date": row["Date"],
            "Time": row["Time"],
            "checkpoint_date": row["checkpoint_date"],
        }
