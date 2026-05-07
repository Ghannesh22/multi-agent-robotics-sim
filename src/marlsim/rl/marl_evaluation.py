from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from marlsim.agents import AgentPolicy
from marlsim.core.actions import Action
from marlsim.core.state import Position
from marlsim.rl.multi_agent_env import MultiAgentRLEnv, MultiAgentObservations
from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.single_agent_env import SingleAgentRLEnv


MARLPolicy = Mapping[str, AgentPolicy | QLearningAgent]


@dataclass(frozen=True)
class MARLEvaluationMetrics:
    """Summary metrics from evaluating multi-agent policies without learning."""

    policy_name: str
    total_episodes: int
    success_count: int
    timeout_count: int
    blocked_move_count: int
    total_reward: float
    total_steps: int
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    final_positions: tuple[dict[str, Position], ...]

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
    def average_reward(self) -> float:
        return self.total_reward / self.total_episodes

    @property
    def average_episode_length(self) -> float:
        return self.total_steps / self.total_episodes


def evaluate_marl_policy(
    env: MultiAgentRLEnv,
    policies: MARLPolicy,
    episodes: int,
    policy_name: str,
) -> MARLEvaluationMetrics:
    """Evaluate multi-agent policies for fixed episodes without Q-table updates."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    _validate_policies(env, policies)

    old_epsilons = {
        agent_id: policy.epsilon
        for agent_id, policy in policies.items()
        if isinstance(policy, QLearningAgent)
    }
    for agent_id in old_epsilons:
        q_policy = policies[agent_id]
        if isinstance(q_policy, QLearningAgent):
            q_policy.epsilon = 0.0

    try:
        return _evaluate_marl_policy(
            env=env,
            policies=policies,
            episodes=episodes,
            policy_name=policy_name,
        )
    finally:
        for agent_id, epsilon in old_epsilons.items():
            q_policy = policies[agent_id]
            if isinstance(q_policy, QLearningAgent):
                q_policy.epsilon = epsilon


def format_marl_evaluation_table(results: list[MARLEvaluationMetrics]) -> str:
    """Format MARL evaluation results as a compact text table."""

    headers = ("policy", "success", "timeouts", "blocked", "avg reward", "avg steps")
    rows = [
        (
            result.policy_name,
            f"{result.success_rate:.2f}",
            f"{result.timeout_frequency:.2f}",
            str(result.blocked_move_count),
            f"{result.average_reward:.2f}",
            f"{result.average_episode_length:.2f}",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    separator = " | ".join("-" * widths[index] for index in range(len(headers)))
    body = [
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _evaluate_marl_policy(
    env: MultiAgentRLEnv,
    policies: MARLPolicy,
    episodes: int,
    policy_name: str,
) -> MARLEvaluationMetrics:
    success_count = 0
    timeout_count = 0
    blocked_move_count = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    final_positions: list[dict[str, Position]] = []

    for _ in range(episodes):
        observations = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_info: dict[str, object] = {}

        while not done:
            actions = _choose_actions(env=env, policies=policies, observations=observations)
            observations, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards.values())
            episode_length += 1
            blocked_move_count += _blocked_count(info)
            final_info = info

        if _all_agents_reached_goal(env, final_info):
            success_count += 1
        if final_info.get("timeout") is True:
            timeout_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_positions.append(
            {
                agent_id: env.env.agents[agent_id].position
                for agent_id in env.controlled_agent_ids
            }
        )

    return MARLEvaluationMetrics(
        policy_name=policy_name,
        total_episodes=episodes,
        success_count=success_count,
        timeout_count=timeout_count,
        blocked_move_count=blocked_move_count,
        total_reward=sum(episode_rewards),
        total_steps=sum(episode_lengths),
        episode_rewards=tuple(episode_rewards),
        episode_lengths=tuple(episode_lengths),
        final_positions=tuple(final_positions),
    )


def _choose_actions(
    env: MultiAgentRLEnv,
    policies: MARLPolicy,
    observations: MultiAgentObservations,
) -> dict[str, int]:
    return {
        agent_id: _choose_action(
            env=env,
            agent_id=agent_id,
            policy=policies[agent_id],
            state=observations[agent_id],
        )
        for agent_id in env.controlled_agent_ids
    }


def _choose_action(
    env: MultiAgentRLEnv,
    agent_id: str,
    policy: AgentPolicy | QLearningAgent,
    state: tuple[int, int, int, int],
) -> int:
    if isinstance(policy, QLearningAgent):
        return _greedy_q_action(policy, state)

    action = policy.choose_action(
        agent=env.env.agents[agent_id],
        agents=env.env.agents,
        obstacles=env.env.config.obstacles,
        width=env.env.config.width,
        height=env.env.config.height,
    )
    return _action_to_int(action)


def _greedy_q_action(policy: QLearningAgent, state: tuple[int, int, int, int]) -> int:
    values = policy.q_table.get(state)
    if values is None:
        return 0
    best_value = max(values)
    for index, value in enumerate(values):
        if value == best_value:
            return index
    raise ValueError("best Q-value not found.")


def _action_to_int(action: Action) -> int:
    try:
        return SingleAgentRLEnv.ACTIONS.index(action)
    except ValueError as error:
        raise ValueError(f"Unsupported action: {action}") from error


def _validate_policies(env: MultiAgentRLEnv, policies: MARLPolicy) -> None:
    expected = set(env.controlled_agent_ids)
    received = set(policies)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ValueError(f"Missing evaluation policies: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected evaluation policies: {sorted(unexpected)}")


def _blocked_count(info: Mapping[str, object]) -> int:
    agent_info = info.get("agents")
    if not isinstance(agent_info, dict):
        return 0
    return sum(
        1
        for details in agent_info.values()
        if isinstance(details, dict) and details.get("blocked") is True
    )


def _all_agents_reached_goal(env: MultiAgentRLEnv, info: Mapping[str, object]) -> bool:
    agent_info = info.get("agents")
    if not isinstance(agent_info, dict):
        return False
    return all(
        isinstance(agent_info.get(agent_id), dict)
        and agent_info[agent_id].get("reached_goal") is True
        for agent_id in env.controlled_agent_ids
    )
