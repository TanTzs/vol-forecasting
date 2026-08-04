"""Forecasting model implementations."""

from .linear_models import HARForecaster, OLSForecaster

__all__ = [
    "CausalBlock",
    "HARForecaster",
    "LSTMForecaster",
    "OLSForecaster",
    "TCNForecaster",
    "load_neural_forecaster",
    "save_neural_forecaster",
]

_NEURAL_EXPORTS = {
    "CausalBlock",
    "LSTMForecaster",
    "TCNForecaster",
    "load_neural_forecaster",
    "save_neural_forecaster",
}


def __getattr__(name: str) -> object:
    """Import PyTorch models only when a neural class is requested."""

    if name in _NEURAL_EXPORTS:
        from . import neural_models

        return getattr(neural_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
