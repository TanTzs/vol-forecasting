"""Train and validate a one-step DQN model selector."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from volatility_forecasting_dqn.rl import (  # noqa: E402
    DQNAgent,
    MODEL_NAMES,
    OneStepEnvironment,
    OneStepStateDataset,
)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed all random sources used by the one-step experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def evaluate(
    environment: OneStepEnvironment,
    agent: DQNAgent,
) -> dict[str, float | int]:
    """Evaluate the greedy policy over one complete environment split."""

    environment.restart()
    rewards = []
    squared_errors = []
    qlike_losses = []
    action_counts = np.zeros(len(MODEL_NAMES), dtype=np.int64)

    while not environment.exhausted:
        state, _ = environment.reset()
        action = agent.select_action(state, explore=False)
        _, reward, _, _, info = environment.step(action)

        rewards.append(reward)
        squared_errors.append(info["selected_mse"])
        qlike_losses.append(info["selected_qlike"])
        action_counts[action] += 1

    metrics: dict[str, float | int] = {
        "mean_reward": float(np.mean(rewards)),
        "mse": float(np.mean(squared_errors)),
        "qlike": float(np.mean(qlike_losses)),
        "n_predictions": len(rewards),
    }
    for model_name, count in zip(MODEL_NAMES, action_counts):
        metrics[f"actions_{model_name}"] = int(count)
    return metrics


def train(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Path]:
    """Run one seeded one-step DQN experiment."""
    if args.train_frequency < 1:
        raise ValueError("train_frequency must be positive")
    if args.training_steps < 1:
        raise ValueError("training_steps must be positive")
    if args.eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    if args.patience < 0:
        raise ValueError("patience cannot be negative")

    configure_reproducibility(args.seed, args.deterministic)

    data_path = (
        args.data_dir / f"one_step_dqn_data_{args.frequency}.csv"
    )
    dataset = OneStepStateDataset.from_csv(data_path)
    train_environment = OneStepEnvironment(
        dataset,
        mode="train",
        reward_metric=args.reward_metric,
        reward_scale=args.reward_scale,
    )
    validation_environment = OneStepEnvironment(
        dataset,
        mode="validation",
        reward_metric=args.reward_metric,
        reward_scale=args.reward_scale,
    )

    agent = DQNAgent(
        device=args.device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
    )

    run_name = (
        f"one_step_{args.frequency}_{args.reward_metric}_seed{args.seed}"
    )
    checkpoint_path = args.checkpoint_dir / f"{run_name}.pt"
    results_path = args.results_dir / f"dqn_training_{run_name}.csv"
    config_path = args.results_dir / f"dqn_config_{run_name}.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "state_dim": agent.state_dim,
            "n_actions": agent.n_actions,
            "train_samples": len(train_environment.sample_indices),
            "validation_samples": len(
                validation_environment.sample_indices
            ),
        }
    )
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(
        f"[{args.frequency}] reward={args.reward_metric}, "
        f"seed={args.seed}, device={agent.device}"
    )
    print(
        f"train samples={len(train_environment.sample_indices):,}, "
        f"validation samples={len(validation_environment.sample_indices):,}"
    )

    history = []
    interval_losses = []
    interval_rewards = []
    best_validation_reward = -np.inf
    evaluations_without_improvement = 0

    for step in range(1, args.training_steps + 1):
        state, _ = train_environment.reset(
            seed=args.seed if step == 1 else None
        )
        action = agent.select_action(state, explore=True)
        next_state, reward, terminated, truncated, _ = (
            train_environment.step(action)
        )
        agent.store_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )
        loss = None
        if step % args.train_frequency == 0:
            loss = agent.update()
        interval_rewards.append(reward)
        if loss is not None:
            interval_losses.append(loss)

        should_evaluate = (
            step % args.eval_interval == 0
            or step == args.training_steps
        )
        if not should_evaluate:
            continue

        validation = evaluate(validation_environment, agent)
        row = {
            "step": step,
            "epsilon": agent.epsilon,
            "replay_size": len(agent.replay_buffer),
            "mean_train_reward": float(np.mean(interval_rewards)),
            "mean_bellman_loss": (
                float(np.mean(interval_losses))
                if interval_losses
                else np.nan
            ),
            **{
                f"validation_{key}": value
                for key, value in validation.items()
            },
        }
        history.append(row)
        interval_rewards.clear()
        interval_losses.clear()

        validation_reward = float(validation["mean_reward"])
        print(
            f"step={step:,} epsilon={agent.epsilon:.4f} "
            f"val_reward={validation_reward:.6f} "
            f"val_mse={validation['mse']:.6f} "
            f"val_qlike={validation['qlike']:.6f}"
        )

        if validation_reward > best_validation_reward:
            best_validation_reward = validation_reward
            evaluations_without_improvement = 0
            agent.save(checkpoint_path)
            print(f"  saved {checkpoint_path.name}")
        else:
            evaluations_without_improvement += 1

        if (
            args.patience > 0
            and evaluations_without_improvement >= args.patience
        ):
            print(
                f"Early stopping after {args.patience} "
                "validation checks without improvement"
            )
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(results_path, index=False)
    print(f"Saved training history to {results_path}")
    return history_frame, checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a one-step DQN model selector."
    )
    parser.add_argument(
        "--frequency",
        required=True,
        choices=["1D", "1H"],
    )
    parser.add_argument(
        "--reward-metric",
        choices=["mse", "qlike"],
        default="qlike",
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1_000,
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=10.0,
    )
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Validation checks without improvement; zero disables stopping.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "dqn",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
