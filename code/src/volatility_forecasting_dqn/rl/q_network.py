"""Dual-branch Q-network for candidate-model selection."""

import torch
from torch import nn

from ..forecasting import CausalBlock
from .one_step_state import LOOKBACK, PREDICTION_COLUMNS


class QNetwork(nn.Module):
    """Estimate one Q-value for each candidate forecasting model.

    The historical branch processes ``[log_rv, six errors, fidelity]`` over
    the lookback window with a causal TCN. The forward branch processes the
    six current candidate forecasts with an MLP.
    """

    def __init__(
        self,
        lookback: int = LOOKBACK,
        n_models: int = len(PREDICTION_COLUMNS),
        num_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 4,
        forward_hidden: int = 32,
    ) -> None:
        super().__init__()
        parameters = {
            "lookback": lookback,
            "n_models": n_models,
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "num_layers": num_layers,
            "forward_hidden": forward_hidden,
        }
        for name, value in parameters.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

        self.lookback = lookback
        self.n_models = n_models
        self.n_temporal_channels = n_models + 2
        self.temporal_size = lookback * self.n_temporal_channels
        self.state_dim = self.temporal_size + n_models
        self.receptive_field = (
            1 + 2 * (kernel_size - 1) * (2**num_layers - 1)
        )

        temporal_layers = []
        in_channels = self.n_temporal_channels
        for layer_index in range(num_layers):
            temporal_layers.append(
                CausalBlock(
                    in_channels=in_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                )
            )
            in_channels = num_channels
        self.temporal_encoder = nn.Sequential(*temporal_layers)

        self.forward_encoder = nn.Sequential(
            nn.Linear(n_models, forward_hidden),
            nn.ReLU(),
        )
        self.output = nn.Linear(num_channels + forward_hidden, n_models)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values without applying softmax."""

        single_state = state.ndim == 1
        if single_state:
            state = state.unsqueeze(0)
        elif state.ndim != 2:
            raise ValueError("state must have shape [state_dim] or [batch, state_dim]")

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"got {state.shape[-1]}"
            )

        temporal = state[:, : self.temporal_size]
        temporal = temporal.reshape(
            -1,
            self.lookback,
            self.n_temporal_channels,
        ).permute(0, 2, 1)
        forward = state[:, self.temporal_size :]

        temporal_features = self.temporal_encoder(temporal)[:, :, -1]
        forward_features = self.forward_encoder(forward)
        q_values = self.output(
            torch.cat([temporal_features, forward_features], dim=1)
        )
        return q_values.squeeze(0) if single_state else q_values
