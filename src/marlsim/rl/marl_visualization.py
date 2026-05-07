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
class MARLTrajectoryStep:
    """One displayed step from a multi-agent policy rollout."""

    step_number: int
    positions: dict[str, Position]
    actions: dict[str, str]
    rewards: dict[str, float]
    blocked: dict[str, bool]
    reached_goals: dict[str, bool]
    done: bool


def rollout_marl_policy(
    env: MultiAgentRLEnv,
    policies: MARLPolicy,
) -> tuple[MARLTrajectoryStep, ...]:
    """Run one episode and return trajectory steps without learning updates."""

    _validate_policies(env, policies)
    old_epsilons = _disable_q_exploration(policies)
    try:
        return _rollout_marl_policy(env=env, policies=policies)
    finally:
        _restore_q_exploration(policies, old_epsilons)


def format_marl_trajectory(trajectory: tuple[MARLTrajectoryStep, ...]) -> str:
    """Format a multi-agent trajectory as readable plain text."""

    if not trajectory:
        return "No trajectory steps recorded."

    lines = ["Trajectory", "----------"]
    for step in trajectory:
        parts = []
        for agent_id in sorted(step.positions):
            position = step.positions[agent_id]
            parts.append(
                (
                    f"{agent_id}: pos=({position.x},{position.y}) "
                    f"action={step.actions[agent_id]} "
                    f"reward={step.rewards[agent_id]:.2f} "
                    f"blocked={step.blocked[agent_id]}"
                )
            )
        lines.append(
            f"step {step.step_number}: "
            + " | ".join(parts)
            + f" | done={step.done}"
        )
    return "\n".join(lines)


def summarize_marl_outcome(
    label: str,
    trajectory: tuple[MARLTrajectoryStep, ...],
) -> str:
    """Return a short success/failure summary for a trajectory."""

    if not trajectory:
        return f"{label}: no rollout steps"

    final_step = trajectory[-1]
    all_at_goal = all(final_step.reached_goals.values())
    outcome = "success" if all_at_goal else "failure"
    return f"{label}: {outcome} in {len(trajectory)} steps"


def _rollout_marl_policy(
    env: MultiAgentRLEnv,
    policies: MARLPolicy,
) -> tuple[MARLTrajectoryStep, ...]:
    observations = env.reset()
    done = False
    trajectory: list[MARLTrajectoryStep] = []

    while not done:
        actions = _choose_actions(env=env, policies=policies, observations=observations)
        observations, rewards, done, info = env.step(actions)
        agent_info = info.get("agents", {})
        trajectory.append(
            MARLTrajectoryStep(
                step_number=env.step_count,
                positions={
                    agent_id: env.env.agents[agent_id].position
                    for agent_id in env.controlled_agent_ids
                },
                actions={
                    agent_id: env.action_to_env_action(actions[agent_id]).value
                    for agent_id in env.controlled_agent_ids
                },
                rewards={
                    agent_id: rewards[agent_id]
                    for agent_id in env.controlled_agent_ids
                },
                blocked={
                    agent_id: (
                        isinstance(agent_info, dict)
                        and isinstance(agent_info.get(agent_id), dict)
                        and agent_info[agent_id].get("blocked") is True
                    )
                    for agent_id in env.controlled_agent_ids
                },
                reached_goals={
                    agent_id: (
                        isinstance(agent_info, dict)
                        and isinstance(agent_info.get(agent_id), dict)
                        and agent_info[agent_id].get("reached_goal") is True
                    )
                    for agent_id in env.controlled_agent_ids
                },
                done=done,
            )
        )

    return tuple(trajectory)


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


def _disable_q_exploration(policies: MARLPolicy) -> dict[str, float]:
    old_epsilons = {}
    for agent_id, policy in policies.items():
        if isinstance(policy, QLearningAgent):
            old_epsilons[agent_id] = policy.epsilon
            policy.epsilon = 0.0
    return old_epsilons


def _restore_q_exploration(policies: MARLPolicy, old_epsilons: Mapping[str, float]) -> None:
    for agent_id, epsilon in old_epsilons.items():
        policy = policies[agent_id]
        if isinstance(policy, QLearningAgent):
            policy.epsilon = epsilon


def _validate_policies(env: MultiAgentRLEnv, policies: MARLPolicy) -> None:
    expected = set(env.controlled_agent_ids)
    received = set(policies)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ValueError(f"Missing rollout policies: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected rollout policies: {sorted(unexpected)}")
