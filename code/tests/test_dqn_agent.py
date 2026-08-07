import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from volatility_forecasting_dqn.rl import DQNAgent, QNetwork  # noqa: E402


def make_network() -> QNetwork:
    return QNetwork(
        lookback=2,
        num_channels=4,
        num_layers=1,
        forward_hidden=4,
    )


class DQNAgentTests(unittest.TestCase):
    def test_greedy_action_uses_largest_q_value(self) -> None:
        network = make_network()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.output.bias.copy_(torch.arange(6, dtype=torch.float32))

        agent = DQNAgent(
            q_network=network,
            batch_size=2,
            replay_capacity=4,
            learning_starts=2,
        )

        action = agent.select_action(np.zeros(22, dtype=np.float32), explore=False)

        self.assertEqual(action, 5)

    def test_epsilon_decays_linearly(self) -> None:
        agent = DQNAgent(
            q_network=make_network(),
            batch_size=2,
            replay_capacity=4,
            learning_starts=2,
            epsilon_start=1.0,
            epsilon_end=0.2,
            epsilon_decay_steps=10,
        )

        self.assertEqual(agent.epsilon, 1.0)
        agent.environment_steps = 5
        self.assertAlmostEqual(agent.epsilon, 0.6)
        agent.environment_steps = 20
        self.assertEqual(agent.epsilon, 0.2)

    def test_terminal_transition_uses_reward_without_bootstrap(self) -> None:
        network = make_network()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()

        agent = DQNAgent(
            q_network=network,
            gamma=0.99,
            batch_size=2,
            replay_capacity=4,
            learning_starts=2,
            target_update_interval=1,
            gradient_clip_norm=None,
        )
        with torch.no_grad():
            agent.target_network.output.bias.fill_(100.0)

        state = np.zeros(22, dtype=np.float32)
        for _ in range(2):
            agent.store_transition(
                state=state,
                action=0,
                reward=-1.0,
                next_state=state,
                terminated=True,
            )

        loss = agent.update()

        self.assertAlmostEqual(loss, 1.0)
        self.assertEqual(agent.gradient_steps, 1)
        for online, target in zip(
            agent.online_network.parameters(),
            agent.target_network.parameters(),
        ):
            torch.testing.assert_close(online, target)

    def test_nonterminal_target_uses_double_dqn_action(self) -> None:
        network = make_network()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.output.bias.copy_(
                torch.tensor([0.0, 5.0, 1.0, 0.0, 0.0, 0.0])
            )

        agent = DQNAgent(
            q_network=network,
            gamma=0.5,
            batch_size=2,
            replay_capacity=4,
            learning_starts=2,
            target_update_interval=10,
            gradient_clip_norm=None,
        )
        with torch.no_grad():
            agent.target_network.output.bias.copy_(
                torch.tensor([0.0, 3.0, 100.0, 0.0, 0.0, 0.0])
            )

        state = np.zeros(22, dtype=np.float32)
        for _ in range(2):
            agent.store_transition(
                state=state,
                action=0,
                reward=0.0,
                next_state=state,
                terminated=False,
            )

        loss = agent.update()

        self.assertAlmostEqual(loss, 2.25)

    def test_update_waits_for_replay_warmup(self) -> None:
        agent = DQNAgent(
            q_network=make_network(),
            batch_size=2,
            replay_capacity=4,
            learning_starts=3,
        )

        self.assertIsNone(agent.update())


if __name__ == "__main__":
    unittest.main()
