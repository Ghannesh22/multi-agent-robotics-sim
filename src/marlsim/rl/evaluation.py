from __future__ import annotations

from dataclasses import dataclass

from marlsim.agents import AgentPolicy
from marlsim.core.actions import Action
from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.single_agent_env import SingleAgentRLEnv


@dataclass(frozen=True)
class EvaluationMetrics:
    """Summary metrics from evaluating one policy without learning updates."""

    policy_name: str
    total_episodes: int
    success_count: int
    timeout_count: int
    total_reward: float
    total_steps: int
    episode_rewards: tuple[float, ...]
    episode_lengths: tuple[int, ...]

    @property
    def failure_count(self) -> int:
        return self.total_episodes - self.success_count

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_episodes

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.total_episodes

    @property
    def average_episode_length(self) -> float:
        return self.total_steps / self.total_episodes

    @property
    def timeout_frequency(self) -> float:
        return self.timeout_count / self.total_episodes


def evaluate_policy(
    env: SingleAgentRLEnv,
    policy: AgentPolicy | QLearningAgent,
    episodes: int,
    policy_name: str,
) -> EvaluationMetrics:
    """Evaluate a policy for fixed episodes without updating learned values."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    old_epsilon: float | None = None
    if isinstance(policy, QLearningAgent):
        old_epsilon = policy.epsilon
        policy.epsilon = 0.0

    try:
        return _evaluate_policy(env=env, policy=policy, episodes=episodes, policy_name=policy_name)
    finally:
        if old_epsilon is not None:
            policy.epsilon = old_epsilon


def format_evaluation_table(results: list[EvaluationMetrics]) -> str:
    """Format policy evaluation results as a small text table."""

    headers = ("policy", "success", "avg reward", "avg steps", "timeouts")
    rows = [
        (
            result.policy_name,
            f"{result.success_rate:.2f}",
            f"{result.average_reward:.2f}",
            f"{result.average_episode_length:.2f}",
            f"{result.timeout_frequency:.2f}",
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


def _evaluate_policy(
    env: SingleAgentRLEnv,
    policy: AgentPolicy | QLearningAgent,
    episodes: int,
    policy_name: str,
) -> EvaluationMetrics:
    success_count = 0
    timeout_count = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_info: dict[str, object] = {}

        while not done:
            action = _choose_action(env=env, policy=policy, state=state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            final_info = info

        if final_info.get("reached_goal") is True:
            success_count += 1
        if final_info.get("timeout") is True:
            timeout_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    total_reward = sum(episode_rewards)
    total_steps = sum(episode_lengths)
    return EvaluationMetrics(
        policy_name=policy_name,
        total_episodes=episodes,
        success_count=success_count,
        timeout_count=timeout_count,
        total_reward=total_reward,
        total_steps=total_steps,
        episode_rewards=tuple(episode_rewards),
        episode_lengths=tuple(episode_lengths),
    )


def _choose_action(
    env: SingleAgentRLEnv,
    policy: AgentPolicy | QLearningAgent,
    state: tuple[int, int, int, int],
) -> int:
    if isinstance(policy, QLearningAgent):
        return policy.choose_action(state)

    agent = env.env.agents[env.controlled_agent_id]
    action = policy.choose_action(
        agent=agent,
        agents=env.env.agents,
        obstacles=env.env.config.obstacles,
        width=env.env.config.width,
        height=env.env.config.height,
    )
    return _action_to_int(action)


def _action_to_int(action: Action) -> int:
    try:
        return SingleAgentRLEnv.ACTIONS.index(action)
    except ValueError as error:
        raise ValueError(f"Unsupported action: {action}") from error
