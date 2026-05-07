from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.single_agent_env import SingleAgentRLEnv


@dataclass(frozen=True)
class TrainingMetrics:
    """Summary metrics from a simple Q-learning training run."""

    total_episodes: int
    success_count: int
    failure_count: int
    timeout_count: int
    total_reward: float
    total_steps: int
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    success_history: tuple[bool, ...]
    timeout_history: tuple[bool, ...]

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.total_episodes

    @property
    def average_episode_length(self) -> float:
        return self.total_steps / self.total_episodes

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_episodes

    @property
    def timeout_frequency(self) -> float:
        return self.timeout_count / self.total_episodes


def train_q_learning(
    env: SingleAgentRLEnv,
    agent: QLearningAgent,
    episodes: int,
) -> TrainingMetrics:
    """Train a QLearningAgent with a simple single-agent episode loop."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    success_count = 0
    timeout_count = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    success_history: list[bool] = []
    timeout_history: list[bool] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_info: dict[str, object] = {}

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1
            final_info = info

        reached_goal = final_info.get("reached_goal") is True
        timed_out = final_info.get("timeout") is True

        if reached_goal:
            success_count += 1
        if timed_out:
            timeout_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_history.append(reached_goal)
        timeout_history.append(timed_out)

    total_reward = sum(episode_rewards)
    total_steps = sum(episode_lengths)
    return TrainingMetrics(
        total_episodes=episodes,
        success_count=success_count,
        failure_count=episodes - success_count,
        timeout_count=timeout_count,
        total_reward=total_reward,
        total_steps=total_steps,
        episode_rewards=tuple(episode_rewards),
        episode_lengths=tuple(episode_lengths),
        success_history=tuple(success_history),
        timeout_history=tuple(timeout_history),
    )


def summarize_training(metrics: TrainingMetrics) -> str:
    """Return a plain text summary of training metrics."""

    return "\n".join(
        (
            "Q-Learning Training Summary",
            "===========================",
            f"Episodes: {metrics.total_episodes}",
            f"Successes: {metrics.success_count}",
            f"Failures: {metrics.failure_count}",
            f"Timeouts: {metrics.timeout_count}",
            f"Success rate: {metrics.success_rate:.2f}",
            f"Timeout frequency: {metrics.timeout_frequency:.2f}",
            f"Average reward: {metrics.average_reward:.2f}",
            f"Average episode length: {metrics.average_episode_length:.2f}",
        )
    )


def reward_chart(metrics: TrainingMetrics, width: int = 30) -> str:
    """Return a dependency-free text chart for per-episode rewards."""

    return _bar_chart("Episode rewards", metrics.episode_rewards, width=width)


def success_rate_chart(metrics: TrainingMetrics, window: int = 10, width: int = 30) -> str:
    """Return a text chart for rolling success rate."""

    if window <= 0:
        raise ValueError("window must be positive.")
    values = []
    for index in range(metrics.total_episodes):
        start = max(0, index + 1 - window)
        current = metrics.success_history[start : index + 1]
        values.append(sum(current) / len(current))
    return _bar_chart(f"Rolling success rate (window={window})", tuple(values), width=width)


def export_training_metrics(metrics: TrainingMetrics, destination: str | Path | TextIO) -> None:
    """Export per-episode training metrics as CSV."""

    close_after_write = False
    if isinstance(destination, str | Path):
        output = Path(destination).open("w", newline="")
        close_after_write = True
    else:
        output = destination

    try:
        writer = csv.writer(output)
        writer.writerow(("episode", "reward", "length", "success", "timeout"))
        for index, (reward, length, success, timeout) in enumerate(
            zip(
                metrics.episode_rewards,
                metrics.episode_lengths,
                metrics.success_history,
                metrics.timeout_history,
                strict=True,
            ),
            start=1,
        ):
            writer.writerow((index, reward, length, int(success), int(timeout)))
    finally:
        if close_after_write:
            output.close()


def _bar_chart(title: str, values: tuple[float, ...], width: int) -> str:
    if width <= 0:
        raise ValueError("width must be positive.")
    if not values:
        return f"{title}\n(no data)"

    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    lines = [title]
    for index, value in enumerate(values, start=1):
        if span == 0:
            bar_length = width if value > 0 else 0
        else:
            bar_length = round(((value - minimum) / span) * width)
        lines.append(f"{index:03d} | {'#' * bar_length} {value:.2f}")
    return "\n".join(lines)
