from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from marlsim.agents import CommunicationAwarePriorityAgent
from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.environment import GridWorldEnv
from marlsim.core.state import Position
from marlsim.rl.communication_q_learning import (
    CommunicationAwareQLearningAgent,
    extend_observation_with_messages,
)
from marlsim.rl.communicating_multi_agent_env import CommunicatingMultiAgentRLEnv
from marlsim.rl.marl_training import (
    MARLRewardMode,
    apply_reward_mode,
    train_independent_q_learning,
)
from marlsim.rl.multi_agent_env import MultiAgentRLEnv
from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.single_agent_env import SingleAgentRLEnv


EnvironmentFactory = Callable[[], GridWorldEnv]


@dataclass(frozen=True)
class CommunicationEvaluationMetrics:
    """Common metrics for comparing communication and no-communication modes."""

    mode_name: str
    total_episodes: int
    success_count: int
    timeout_count: int
    blocked_move_count: int
    total_reward: float
    total_steps: int
    communication_usage_count: int = 0
    predicted_conflict_count: int = 0
    waiting_count: int = 0

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

    @property
    def predicted_conflict_frequency(self) -> float:
        if self.communication_usage_count == 0:
            return 0.0
        return self.predicted_conflict_count / self.communication_usage_count

    @property
    def waiting_frequency(self) -> float:
        if self.communication_usage_count == 0:
            return 0.0
        return self.waiting_count / self.communication_usage_count


def evaluate_independent_marl(
    env_factory: EnvironmentFactory,
    episodes: int,
    seed_offset: int = 0,
) -> CommunicationEvaluationMetrics:
    """Train independent MARL without communication and return comparable metrics."""

    env = MultiAgentRLEnv(env_factory(), ("agent_1", "agent_2"))
    agents = {
        agent_id: QLearningAgent(
            action_size=env.action_size,
            epsilon=0.2,
            seed=seed_offset + index,
        )
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }
    metrics = train_independent_q_learning(env=env, agents=agents, episodes=episodes)
    return CommunicationEvaluationMetrics(
        mode_name="independent",
        total_episodes=metrics.total_episodes,
        success_count=metrics.success_count,
        timeout_count=metrics.timeout_count,
        blocked_move_count=metrics.blocked_move_count,
        total_reward=metrics.total_reward,
        total_steps=metrics.total_steps,
    )


def evaluate_rule_based_communication(
    env_factory: EnvironmentFactory,
    episodes: int,
) -> CommunicationEvaluationMetrics:
    """Evaluate deterministic rule-based communication coordination."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    success_count = 0
    timeout_count = 0
    blocked_move_count = 0
    total_reward = 0.0
    total_steps = 0
    communication_usage_count = 0
    predicted_conflict_count = 0
    waiting_count = 0

    for _ in range(episodes):
        env = CommunicatingMultiAgentRLEnv(env_factory(), ("agent_1", "agent_2"))
        env.reset()
        policy = CommunicationAwarePriorityAgent()
        done = False
        final_info: dict[str, object] = {}

        while not done:
            messages = _build_rule_based_messages(env=env, policy=policy)
            message_counts = _message_counts(env, messages)
            communication_usage_count += message_counts["communication"]
            predicted_conflict_count += message_counts["predicted_conflict"]
            waiting_count += message_counts["waiting"]

            actions = {
                agent_id: _action_to_int(
                    policy.choose_action_with_messages(
                        agent=agent,
                        agents=env.env.agents,
                        obstacles=env.env.config.obstacles,
                        width=env.env.config.width,
                        height=env.env.config.height,
                        messages=messages,
                    )
                )
                for agent_id, agent in env.env.agents.items()
                if agent_id in env.controlled_agent_ids
            }
            _, rewards, done, info = env.step(actions, messages)
            total_reward += sum(rewards.values())
            total_steps += 1
            blocked_move_count += _blocked_count(info)
            final_info = info

        if _all_agents_reached_goal(env, final_info):
            success_count += 1
        if final_info.get("timeout") is True:
            timeout_count += 1

    return CommunicationEvaluationMetrics(
        mode_name="rule-based communication",
        total_episodes=episodes,
        success_count=success_count,
        timeout_count=timeout_count,
        blocked_move_count=blocked_move_count,
        total_reward=total_reward,
        total_steps=total_steps,
        communication_usage_count=communication_usage_count,
        predicted_conflict_count=predicted_conflict_count,
        waiting_count=waiting_count,
    )


def evaluate_communication_q_learning(
    env_factory: EnvironmentFactory,
    episodes: int,
    seed_offset: int = 0,
    reward_mode: MARLRewardMode | str = MARLRewardMode.INDIVIDUAL,
) -> CommunicationEvaluationMetrics:
    """Train communication-aware tabular Q-learning and collect message metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    reward_mode = MARLRewardMode(reward_mode)

    env = CommunicatingMultiAgentRLEnv(env_factory(), ("agent_1", "agent_2"))
    agents = {
        agent_id: CommunicationAwareQLearningAgent(
            action_size=env.action_size,
            epsilon=0.2,
            seed=seed_offset + index,
        )
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }

    success_count = 0
    timeout_count = 0
    blocked_move_count = 0
    total_reward = 0.0
    total_steps = 0
    communication_usage_count = 0
    predicted_conflict_count = 0
    waiting_count = 0

    for _ in range(episodes):
        observations = env.reset()
        done = False
        final_info: dict[str, object] = {}

        while not done:
            messages = _build_rule_based_messages(env=env, policy=CommunicationAwarePriorityAgent())
            message_counts = _message_counts(env, messages)
            communication_usage_count += message_counts["communication"]
            predicted_conflict_count += message_counts["predicted_conflict"]
            waiting_count += message_counts["waiting"]

            communication_observations = extend_observation_with_messages(
                observations,
                agents=env.env.agents,
                messages=messages,
            )
            actions = {
                agent_id: agents[agent_id].choose_action(
                    communication_observations[agent_id]
                )
                for agent_id in env.controlled_agent_ids
            }

            next_observations, env_rewards, done, info = env.step(actions, messages)
            next_messages = {} if done else _build_rule_based_messages(
                env=env,
                policy=CommunicationAwarePriorityAgent(),
            )
            next_communication_observations = extend_observation_with_messages(
                next_observations,
                agents=env.env.agents,
                messages=next_messages,
            )
            rewards = apply_reward_mode(
                env_rewards,
                info=info,
                controlled_agent_ids=env.controlled_agent_ids,
                reward_mode=reward_mode,
            )

            for agent_id in env.controlled_agent_ids:
                agents[agent_id].update(
                    state=communication_observations[agent_id],
                    action=actions[agent_id],
                    reward=rewards[agent_id],
                    next_state=next_communication_observations[agent_id],
                    done=done,
                )

            observations = next_observations
            total_reward += sum(rewards.values())
            total_steps += 1
            blocked_move_count += _blocked_count(info)
            final_info = info

        if _all_agents_reached_goal(env, final_info):
            success_count += 1
        if final_info.get("timeout") is True:
            timeout_count += 1

    return CommunicationEvaluationMetrics(
        mode_name="communication q-learning",
        total_episodes=episodes,
        success_count=success_count,
        timeout_count=timeout_count,
        blocked_move_count=blocked_move_count,
        total_reward=total_reward,
        total_steps=total_steps,
        communication_usage_count=communication_usage_count,
        predicted_conflict_count=predicted_conflict_count,
        waiting_count=waiting_count,
    )


def compare_communication_modes(
    env_factory: EnvironmentFactory,
    episodes: int = 50,
) -> tuple[CommunicationEvaluationMetrics, ...]:
    """Compare no communication, rule-based communication, and communication-aware Q-learning."""

    return (
        evaluate_independent_marl(env_factory, episodes, seed_offset=0),
        evaluate_rule_based_communication(env_factory, episodes),
        evaluate_communication_q_learning(env_factory, episodes, seed_offset=100),
    )


def summarize_communication_metrics(metrics: CommunicationEvaluationMetrics) -> str:
    """Return a compact text summary for one communication evaluation result."""

    return "\n".join(
        (
            f"{metrics.mode_name} Communication Evaluation",
            "=" * (len(metrics.mode_name) + 25),
            f"Episodes: {metrics.total_episodes}",
            f"Success rate: {metrics.success_rate:.2f}",
            f"Timeout frequency: {metrics.timeout_frequency:.2f}",
            f"Blocked moves: {metrics.blocked_move_count}",
            f"Average reward: {metrics.average_reward:.2f}",
            f"Average episode length: {metrics.average_episode_length:.2f}",
            f"Communication messages: {metrics.communication_usage_count}",
            f"Predicted conflict frequency: {metrics.predicted_conflict_frequency:.2f}",
            f"Waiting frequency: {metrics.waiting_frequency:.2f}",
        )
    )


def format_communication_table(
    results: tuple[CommunicationEvaluationMetrics, ...],
) -> str:
    """Format communication comparison metrics as a readable table."""

    headers = (
        "mode",
        "success",
        "timeouts",
        "blocked",
        "avg reward",
        "avg steps",
        "messages",
        "conflict freq",
        "waiting freq",
    )
    rows = [
        (
            result.mode_name,
            f"{result.success_rate:.2f}",
            f"{result.timeout_frequency:.2f}",
            str(result.blocked_move_count),
            f"{result.average_reward:.2f}",
            f"{result.average_episode_length:.2f}",
            str(result.communication_usage_count),
            f"{result.predicted_conflict_frequency:.2f}",
            f"{result.waiting_frequency:.2f}",
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


def _build_rule_based_messages(
    *,
    env: CommunicatingMultiAgentRLEnv,
    policy: CommunicationAwarePriorityAgent,
) -> dict[str, CommunicationMessage]:
    return {
        agent_id: policy.build_message(
            agent=agent,
            agents=env.env.agents,
            obstacles=env.env.config.obstacles,
            width=env.env.config.width,
            height=env.env.config.height,
            step_count=env.step_count,
        )
        for agent_id, agent in env.env.agents.items()
        if agent_id in env.controlled_agent_ids
    }


def _message_counts(
    env: CommunicatingMultiAgentRLEnv,
    messages: Mapping[str, CommunicationMessage],
) -> dict[str, int]:
    observations = env.get_observations()
    communication_observations = extend_observation_with_messages(
        observations,
        agents=env.env.agents,
        messages=messages,
    )
    return {
        "communication": len(messages),
        "predicted_conflict": sum(
            observation[6] for observation in communication_observations.values()
        ),
        "waiting": sum(1 for message in messages.values() if message.waiting),
    }


def _blocked_count(info: Mapping[str, object]) -> int:
    agent_info = info.get("agents")
    if not isinstance(agent_info, dict):
        return 0
    return sum(
        1
        for details in agent_info.values()
        if isinstance(details, dict) and details.get("blocked") is True
    )


def _all_agents_reached_goal(
    env: CommunicatingMultiAgentRLEnv,
    info: Mapping[str, object],
) -> bool:
    agent_info = info.get("agents")
    if not isinstance(agent_info, dict):
        return False
    return all(
        isinstance(agent_info.get(agent_id), dict)
        and agent_info[agent_id].get("reached_goal") is True
        for agent_id in env.controlled_agent_ids
    )


def _action_to_int(action: Action) -> int:
    try:
        return SingleAgentRLEnv.ACTIONS.index(action)
    except ValueError as error:
        raise ValueError(f"Unsupported action: {action}") from error
