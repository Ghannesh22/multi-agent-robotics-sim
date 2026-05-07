from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from marlsim.rl.multi_agent_env import MultiAgentRLEnv, MultiAgentObservations
from marlsim.rl.q_learning import QLearningAgent


@dataclass(frozen=True)
class MARLTrainingMetrics:
    """Summary metrics from independent multi-agent Q-learning."""

    total_episodes: int
    success_count: int
    timeout_count: int
    blocked_move_count: int
    total_reward: float
    total_steps: int
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    success_history: tuple[bool, ...]
    timeout_history: tuple[bool, ...]
    per_agent_rewards: dict[str, tuple[float, ...]]

    @property
    def failure_count(self) -> int:
        return self.total_episodes - self.success_count

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_episodes

    @property
    def timeout_frequency(self) -> float:
        return self.timeout_count / self.total_episodes

    @property
    def average_episode_length(self) -> float:
        return self.total_steps / self.total_episodes

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.total_episodes


class MARLRewardMode(str, Enum):
    """Reward modes for independent MARL training."""

    INDIVIDUAL = "individual"
    SHARED = "shared"


def train_independent_q_learning(
    env: MultiAgentRLEnv,
    agents: Mapping[str, QLearningAgent],
    episodes: int,
    reward_mode: MARLRewardMode | str = MARLRewardMode.INDIVIDUAL,
) -> MARLTrainingMetrics:
    """Train one independent Q-learning agent per controlled environment agent."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    _validate_agents(env, agents)
    reward_mode = MARLRewardMode(reward_mode)

    success_count = 0
    timeout_count = 0
    blocked_move_count = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    success_history: list[bool] = []
    timeout_history: list[bool] = []
    per_agent_rewards: dict[str, list[float]] = {
        agent_id: [] for agent_id in env.controlled_agent_ids
    }

    for _ in range(episodes):
        observations = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_agent_rewards = {agent_id: 0.0 for agent_id in env.controlled_agent_ids}
        final_info: dict[str, object] = {}

        while not done:
            actions = {
                agent_id: agents[agent_id].choose_action(observations[agent_id])
                for agent_id in env.controlled_agent_ids
            }
            next_observations, env_rewards, done, info = env.step(actions)
            rewards = apply_reward_mode(
                env_rewards,
                info=info,
                controlled_agent_ids=env.controlled_agent_ids,
                reward_mode=reward_mode,
            )

            for agent_id in env.controlled_agent_ids:
                agents[agent_id].update(
                    state=observations[agent_id],
                    action=actions[agent_id],
                    reward=rewards[agent_id],
                    next_state=next_observations[agent_id],
                    done=done,
                )
                episode_agent_rewards[agent_id] += rewards[agent_id]

            observations = next_observations
            episode_reward += sum(rewards.values())
            episode_length += 1
            blocked_move_count += _blocked_count(info)
            final_info = info

        reached_all_goals = _all_agents_reached_goal(env, final_info)
        timed_out = final_info.get("timeout") is True
        if reached_all_goals:
            success_count += 1
        if timed_out:
            timeout_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_history.append(reached_all_goals)
        timeout_history.append(timed_out)
        for agent_id, reward in episode_agent_rewards.items():
            per_agent_rewards[agent_id].append(reward)

    total_reward = sum(episode_rewards)
    total_steps = sum(episode_lengths)
    return MARLTrainingMetrics(
        total_episodes=episodes,
        success_count=success_count,
        timeout_count=timeout_count,
        blocked_move_count=blocked_move_count,
        total_reward=total_reward,
        total_steps=total_steps,
        episode_rewards=tuple(episode_rewards),
        episode_lengths=tuple(episode_lengths),
        success_history=tuple(success_history),
        timeout_history=tuple(timeout_history),
        per_agent_rewards={
            agent_id: tuple(rewards)
            for agent_id, rewards in per_agent_rewards.items()
        },
    )


def summarize_marl_training(metrics: MARLTrainingMetrics) -> str:
    """Return a plain text summary of independent MARL training."""

    return "\n".join(
        (
            "Independent Q-Learning MARL Summary",
            "===================================",
            f"Episodes: {metrics.total_episodes}",
            f"Successes: {metrics.success_count}",
            f"Failures: {metrics.failure_count}",
            f"Timeouts: {metrics.timeout_count}",
            f"Blocked moves: {metrics.blocked_move_count}",
            f"Success rate: {metrics.success_rate:.2f}",
            f"Timeout frequency: {metrics.timeout_frequency:.2f}",
            f"Average reward: {metrics.average_reward:.2f}",
            f"Average episode length: {metrics.average_episode_length:.2f}",
        )
    )


def apply_reward_mode(
    rewards: Mapping[str, float],
    *,
    info: Mapping[str, object],
    controlled_agent_ids: tuple[str, ...],
    reward_mode: MARLRewardMode | str,
) -> dict[str, float]:
    """Return per-agent training rewards for the selected MARL reward mode."""

    mode = MARLRewardMode(reward_mode)
    if mode == MARLRewardMode.INDIVIDUAL:
        return {agent_id: rewards[agent_id] for agent_id in controlled_agent_ids}

    team_reward = _team_reward(rewards, info=info, controlled_agent_ids=controlled_agent_ids)
    return {agent_id: team_reward for agent_id in controlled_agent_ids}


def _validate_agents(env: MultiAgentRLEnv, agents: Mapping[str, QLearningAgent]) -> None:
    expected = set(env.controlled_agent_ids)
    received = set(agents)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ValueError(f"Missing Q-learning agents: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected Q-learning agents: {sorted(unexpected)}")


def _blocked_count(info: Mapping[str, object]) -> int:
    agent_info = info.get("agents")
    if not isinstance(agent_info, dict):
        return 0
    return sum(
        1
        for details in agent_info.values()
        if isinstance(details, dict) and details.get("blocked") is True
    )


def _team_reward(
    rewards: Mapping[str, float],
    *,
    info: Mapping[str, object],
    controlled_agent_ids: tuple[str, ...],
) -> float:
    # Keep the first team reward simple: average individual learning signal plus
    # an extra group bonus when everyone reaches their goal together.
    team_reward = sum(rewards[agent_id] for agent_id in controlled_agent_ids) / len(
        controlled_agent_ids
    )
    if _all_agents_reached_goal_ids(controlled_agent_ids, info):
        team_reward += 5.0
    return team_reward


def _all_agents_reached_goal(env: MultiAgentRLEnv, info: Mapping[str, object]) -> bool:
    return _all_agents_reached_goal_ids(env.controlled_agent_ids, info)


def _all_agents_reached_goal_ids(
    controlled_agent_ids: tuple[str, ...], info: Mapping[str, object]
) -> bool:
    agent_info = info.get("agents")
    if not isinstance(agent_info, dict):
        return False
    return all(
        isinstance(agent_info.get(agent_id), dict)
        and agent_info[agent_id].get("reached_goal") is True
        for agent_id in controlled_agent_ids
    )
