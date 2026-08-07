"""Deep Q-learning agent and replay buffer."""

from copy import deepcopy
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .q_network import QNetwork


class ReplayBuffer:
    """Fixed-size replay memory backed by NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        seed: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if state_dim < 1:
            raise ValueError("state_dim must be positive")

        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty(
            (capacity, state_dim),
            dtype=np.float32,
        )
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        expected_shape = (self.state_dim,)
        if state.shape != expected_shape or next_state.shape != expected_shape:
            raise ValueError(
                f"state and next_state must have shape {expected_shape}"
            )
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError("states must contain only finite values")
        if not isinstance(action, Integral) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        self.states[self.position] = state
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position] = bool(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        if batch_size > self.size:
            raise ValueError("batch_size exceeds the number of stored transitions")

        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


class DQNAgent:
    """Double-DQN agent with epsilon-greedy exploration."""

    def __init__(
        self,
        q_network: QNetwork | None = None,
        *,
        device: str = "auto",
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        replay_capacity: int = 50_000,
        learning_starts: int = 1_000,
        target_update_interval: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        gradient_clip_norm: float | None = 10.0,
        seed: int = 42,
        loss_function: nn.Module | None = None,
    ) -> None:
        self._validate_settings(
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            gradient_clip_norm=gradient_clip_norm,
        )
        self._set_seed(seed)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.online_network = q_network if q_network is not None else QNetwork()
        self.target_network = deepcopy(self.online_network)
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.state_dim = self.online_network.state_dim
        self.n_actions = self.online_network.n_models
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gradient_clip_norm = gradient_clip_norm

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )
        self.loss_function = (
            loss_function if loss_function is not None else nn.MSELoss()
        )
        self.rng = np.random.default_rng(seed)
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            state_dim=self.state_dim,
            seed=seed,
        )
        self.environment_steps = 0
        self.gradient_steps = 0

    @staticmethod
    def _validate_settings(**settings: object) -> None:
        if settings["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= settings["gamma"] <= 1:
            raise ValueError("gamma must be between zero and one")
        for name in [
            "batch_size",
            "replay_capacity",
            "learning_starts",
            "target_update_interval",
            "epsilon_decay_steps",
        ]:
            if settings[name] < 1:
                raise ValueError(f"{name} must be positive")
        if settings["replay_capacity"] < settings["batch_size"]:
            raise ValueError("replay_capacity must be at least batch_size")
        if settings["learning_starts"] > settings["replay_capacity"]:
            raise ValueError("learning_starts cannot exceed replay_capacity")
        epsilon_start = settings["epsilon_start"]
        epsilon_end = settings["epsilon_end"]
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError(
                "epsilon values must satisfy "
                "0 <= epsilon_end <= epsilon_start <= 1"
            )
        gradient_clip_norm = settings["gradient_clip_norm"]
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def epsilon(self) -> float:
        if self.environment_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        progress = min(
            self.environment_steps / self.epsilon_decay_steps,
            1.0,
        )
        return self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

    def select_action(
        self,
        state: np.ndarray | torch.Tensor,
        explore: bool = True,
    ) -> int:
        """Select one action using epsilon-greedy exploration."""

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )
        was_training = self.online_network.training
        self.online_network.eval()
        with torch.no_grad():
            action = int(self.online_network(state_tensor).argmax().item())
        self.online_network.train(was_training)
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Add one Gymnasium transition to replay memory."""

        if not isinstance(action, Integral) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be between 0 and {self.n_actions - 1}")
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated or truncated,
        )
        self.environment_steps += 1

    def update(self) -> float | None:
        """Run one Double-DQN update when replay warm-up is complete."""

        required_samples = max(self.batch_size, self.learning_starts)
        if len(self.replay_buffer) < required_samples:
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size, self.device)
        )

        current_q = self.online_network(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(
                dim=1,
                keepdim=True,
            )
            next_q = self.target_network(next_states).gather(
                1,
                next_actions,
            ).squeeze(1)
            targets = rewards + self.gamma * (~dones).float() * next_q

        loss = self.loss_function(current_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                self.gradient_clip_norm,
            )
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_interval == 0:
            self.sync_target_network()
        return float(loss.item())

    def sync_target_network(self) -> None:
        """Copy all online-network parameters to the target network."""

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def save(self, path: str | Path) -> None:
        """Save trainable agent state, excluding replay memory."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "environment_steps": self.environment_steps,
                "gradient_steps": self.gradient_steps,
            },
            path,
        )

    def load(
        self,
        path: str | Path,
        load_optimizer: bool = True,
    ) -> None:
        """Restore a checkpoint created by :meth:`save`."""

        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = int(checkpoint["environment_steps"])
        self.gradient_steps = int(checkpoint["gradient_steps"])
