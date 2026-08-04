import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from volatility_forecasting_dqn.rl import QNetwork  # noqa: E402


class QNetworkTests(unittest.TestCase):
    def test_default_batch_shape_and_receptive_field(self) -> None:
        network = QNetwork()
        states = torch.randn(4, 486)

        q_values = network(states)

        self.assertEqual(q_values.shape, (4, 6))
        self.assertEqual(network.receptive_field, 61)

    def test_single_state_shape(self) -> None:
        network = QNetwork()

        q_values = network(torch.randn(486))

        self.assertEqual(q_values.shape, (6,))

    def test_gradients_flow_through_both_branches(self) -> None:
        network = QNetwork()
        q_values = network(torch.randn(3, 486))

        q_values.square().mean().backward()

        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in network.parameters()
            )
        )

    def test_wrong_state_dimension_is_rejected(self) -> None:
        network = QNetwork()

        with self.assertRaisesRegex(ValueError, "Expected state dimension 486"):
            network(torch.randn(2, 485))


if __name__ == "__main__":
    unittest.main()
