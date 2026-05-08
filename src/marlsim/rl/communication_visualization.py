from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from marlsim.agents import CommunicationAwarePriorityAgent
from marlsim.communication import CommunicationMessage, format_message
from marlsim.core.actions import Action
from marlsim.core.state import Position
from marlsim.rl.communication_q_learning import (
    CommunicationAwareQLearningAgent,
    extend_observation_with_messages,
)
from marlsim.rl.communicating_multi_agent_env import CommunicatingMultiAgentRLEnv
from marlsim.rl.multi_agent_env import MultiAgentRLEnv, MultiAgentObservations
from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.single_agent_env import SingleAgentRLEnv


@dataclass(frozen=True)
class CommunicationTrajectoryStep:
    """One rendered step from a communication-aware rollout."""

    step_number: int
    mode_name: str
    positions: dict[str, Position]
    actions: dict[str, str]
    rewards: dict[str, float]
    messages: dict[str, CommunicationMessage]
    formatted_messages: tuple[str, ...]
    waiting: dict[str, bool]
    priorities: dict[str, int]
    predicted_conflicts: dict[str, bool]
    blocked: dict[str, bool]
    reached_goals: dict[str, bool]
    done: bool
    timeout: bool


def rollout_independent_communication_trace(
    env: MultiAgentRLEnv,
    agents: Mapping[str, QLearningAgent],
    mode_name: str = "independent MARL",
) -> tuple[CommunicationTrajectoryStep, ...]:
    """Roll out independent Q-learning agents with no communication messages."""

    _validate_agents(env.controlled_agent_ids, agents)
    old_epsilons = _disable_q_exploration(agents)
    try:
        observations = env.reset()
        done = False
        trajectory: list[CommunicationTrajectoryStep] = []

        while not done:
            actions = {
                agent_id: agents[agent_id].choose_action(observations[agent_id])
                for agent_id in env.controlled_agent_ids
            }
            observations, rewards, done, info = env.step(actions)
            trajectory.append(
                _build_step(
                    env=env,
                    mode_name=mode_name,
                    actions=actions,
                    rewards=rewards,
                    messages={},
                    predicted_conflicts={agent_id: False for agent_id in env.controlled_agent_ids},
                    info=info,
                    done=done,
                )
            )

        return tuple(trajectory)
    finally:
        _restore_q_exploration(agents, old_epsilons)


def rollout_rule_based_communication_trace(
    env: CommunicatingMultiAgentRLEnv,
    mode_name: str = "rule-based communication",
) -> tuple[CommunicationTrajectoryStep, ...]:
    """Roll out deterministic message-aware priority coordination."""

    policy = CommunicationAwarePriorityAgent()
    env.reset()
    done = False
    trajectory: list[CommunicationTrajectoryStep] = []

    while not done:
        messages = _build_messages(env=env, policy=policy)
        predicted_conflicts = _predicted_conflicts(env=env, messages=messages)
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
        trajectory.append(
            _build_step(
                env=env,
                mode_name=mode_name,
                actions=actions,
                rewards=rewards,
                messages=messages,
                predicted_conflicts=predicted_conflicts,
                info=info,
                done=done,
            )
        )

    return tuple(trajectory)


def rollout_communication_q_learning_trace(
    env: CommunicatingMultiAgentRLEnv,
    agents: Mapping[str, CommunicationAwareQLearningAgent],
    mode_name: str = "communication-aware Q-learning",
) -> tuple[CommunicationTrajectoryStep, ...]:
    """Roll out communication-aware Q-learning agents without updates."""

    _validate_agents(env.controlled_agent_ids, agents)
    old_epsilons = _disable_q_exploration(agents)
    try:
        observations = env.reset()
        done = False
        trajectory: list[CommunicationTrajectoryStep] = []

        while not done:
            policy = CommunicationAwarePriorityAgent()
            messages = _build_messages(env=env, policy=policy)
            predicted_conflicts = _predicted_conflicts(env=env, messages=messages)
            communication_observations = extend_observation_with_messages(
                observations,
                agents=env.env.agents,
                messages=messages,
            )
            actions = {
                agent_id: agents[agent_id].choose_action(communication_observations[agent_id])
                for agent_id in env.controlled_agent_ids
            }
            observations, rewards, done, info = env.step(actions, messages)
            trajectory.append(
                _build_step(
                    env=env,
                    mode_name=mode_name,
                    actions=actions,
                    rewards=rewards,
                    messages=messages,
                    predicted_conflicts=predicted_conflicts,
                    info=info,
                    done=done,
                )
            )

        return tuple(trajectory)
    finally:
        _restore_q_exploration(agents, old_epsilons)


def format_communication_rollout(
    trajectory: tuple[CommunicationTrajectoryStep, ...],
) -> str:
    """Format a communication trajectory as readable plain text."""

    if not trajectory:
        return "No communication trajectory steps recorded."

    lines = [
        f"Communication Rollout: {trajectory[0].mode_name}",
        "------------------------------------------",
    ]
    for step in trajectory:
        lines.append(
            f"STEP {step.step_number} done={step.done} timeout={step.timeout}"
        )
        for agent_id in sorted(step.positions):
            position = step.positions[agent_id]
            message = step.messages.get(agent_id)
            sent = format_message(message) if message is not None else "none"
            lines.extend(
                (
                    f"{agent_id}:",
                    f"  pos=({position.x},{position.y})",
                    f"  action={step.actions[agent_id]}",
                    f"  reward={step.rewards[agent_id]:.2f}",
                    f"  blocked={step.blocked[agent_id]}",
                    f"  reached_goal={step.reached_goals[agent_id]}",
                    f"  sent=\"{sent}\"",
                    f"  waiting={step.waiting[agent_id]}",
                    f"  priority={step.priorities[agent_id]}",
                    f"  predicted_conflict={step.predicted_conflicts[agent_id]}",
                )
            )
    return "\n".join(lines)


def format_message_timeline(
    trajectory: tuple[CommunicationTrajectoryStep, ...],
) -> str:
    """Format only the messages from a communication trajectory."""

    if not trajectory:
        return "No messages recorded."

    lines = ["Message Timeline", "----------------"]
    for step in trajectory:
        if not step.formatted_messages:
            lines.append(f"STEP {step.step_number}: no messages")
            continue
        lines.append(f"STEP {step.step_number}:")
        lines.extend(f"- {message}" for message in step.formatted_messages)
    return "\n".join(lines)


def summarize_communication_episode(
    label: str,
    trajectory: tuple[CommunicationTrajectoryStep, ...],
) -> str:
    """Summarize outcome and communication behavior for one rollout."""

    if not trajectory:
        return f"{label}: no rollout steps"

    final_step = trajectory[-1]
    success = all(final_step.reached_goals.values())
    blocked_moves = sum(
        1
        for step in trajectory
        for blocked in step.blocked.values()
        if blocked
    )
    message_count = sum(len(step.messages) for step in trajectory)
    predicted_conflicts = sum(
        1
        for step in trajectory
        for predicted in step.predicted_conflicts.values()
        if predicted
    )
    waiting_count = sum(
        1
        for step in trajectory
        for waiting in step.waiting.values()
        if waiting
    )
    prevented_conflicts = sum(
        1
        for step in trajectory
        if any(step.predicted_conflicts.values()) and not any(step.blocked.values())
    )
    unnecessary_waiting = sum(
        1
        for step in trajectory
        for agent_id, waiting in step.waiting.items()
        if waiting
        and not step.predicted_conflicts[agent_id]
        and not step.reached_goals[agent_id]
    )
    outcome = "success" if success else "failure"
    return (
        f"{label}: {outcome} in {len(trajectory)} steps | "
        f"blocked={blocked_moves} | messages={message_count} | "
        f"predicted_conflicts={predicted_conflicts} | waiting={waiting_count} | "
        f"conflict-free predicted-conflict steps={prevented_conflicts} | "
        f"possible unnecessary waits={unnecessary_waiting}"
    )


def visualize_waiting_behavior(
    trajectory: tuple[CommunicationTrajectoryStep, ...],
) -> str:
    """Return a compact waiting trace for each step."""

    if not trajectory:
        return "No waiting behavior recorded."

    lines = ["Waiting Behavior", "----------------"]
    for step in trajectory:
        waiting_agents = [
            agent_id for agent_id, waiting in sorted(step.waiting.items()) if waiting
        ]
        if waiting_agents:
            lines.append(f"STEP {step.step_number}: waiting={', '.join(waiting_agents)}")
        else:
            lines.append(f"STEP {step.step_number}: waiting=none")
    return "\n".join(lines)


def visualize_conflict_predictions(
    trajectory: tuple[CommunicationTrajectoryStep, ...],
) -> str:
    """Return a compact predicted-conflict trace for each step."""

    if not trajectory:
        return "No conflict predictions recorded."

    lines = ["Conflict Predictions", "--------------------"]
    for step in trajectory:
        predicted_agents = [
            agent_id
            for agent_id, predicted in sorted(step.predicted_conflicts.items())
            if predicted
        ]
        blocked_agents = [
            agent_id for agent_id, blocked in sorted(step.blocked.items()) if blocked
        ]
        lines.append(
            f"STEP {step.step_number}: predicted={_format_agent_list(predicted_agents)} "
            f"blocked={_format_agent_list(blocked_agents)}"
        )
    return "\n".join(lines)


def _build_step(
    *,
    env: MultiAgentRLEnv | CommunicatingMultiAgentRLEnv,
    mode_name: str,
    actions: Mapping[str, int],
    rewards: Mapping[str, float],
    messages: Mapping[str, CommunicationMessage],
    predicted_conflicts: Mapping[str, bool],
    info: Mapping[str, object],
    done: bool,
) -> CommunicationTrajectoryStep:
    agent_info = info.get("agents", {})
    return CommunicationTrajectoryStep(
        step_number=env.step_count,
        mode_name=mode_name,
        positions={
            agent_id: env.env.agents[agent_id].position
            for agent_id in env.controlled_agent_ids
        },
        actions={
            agent_id: env.action_to_env_action(actions[agent_id]).value
            for agent_id in env.controlled_agent_ids
        },
        rewards={agent_id: rewards[agent_id] for agent_id in env.controlled_agent_ids},
        messages=dict(messages),
        formatted_messages=tuple(format_message(message) for message in messages.values()),
        waiting={
            agent_id: messages.get(agent_id).waiting if agent_id in messages else False
            for agent_id in env.controlled_agent_ids
        },
        priorities={
            agent_id: messages.get(agent_id).priority if agent_id in messages else -1
            for agent_id in env.controlled_agent_ids
        },
        predicted_conflicts={
            agent_id: predicted_conflicts.get(agent_id, False)
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
        timeout=info.get("timeout") is True,
    )


def _build_messages(
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


def _predicted_conflicts(
    *,
    env: CommunicatingMultiAgentRLEnv,
    messages: Mapping[str, CommunicationMessage],
) -> dict[str, bool]:
    observations = env.get_observations()
    extended = extend_observation_with_messages(
        observations,
        agents=env.env.agents,
        messages=messages,
    )
    return {
        agent_id: bool(observation[6])
        for agent_id, observation in extended.items()
    }


def _action_to_int(action: Action) -> int:
    try:
        return SingleAgentRLEnv.ACTIONS.index(action)
    except ValueError as error:
        raise ValueError(f"Unsupported action: {action}") from error


def _disable_q_exploration(
    agents: Mapping[str, QLearningAgent],
) -> dict[str, float]:
    old_epsilons = {}
    for agent_id, agent in agents.items():
        old_epsilons[agent_id] = agent.epsilon
        agent.epsilon = 0.0
    return old_epsilons


def _restore_q_exploration(
    agents: Mapping[str, QLearningAgent],
    old_epsilons: Mapping[str, float],
) -> None:
    for agent_id, epsilon in old_epsilons.items():
        agents[agent_id].epsilon = epsilon


def _validate_agents(
    controlled_agent_ids: tuple[str, ...],
    agents: Mapping[str, QLearningAgent],
) -> None:
    expected = set(controlled_agent_ids)
    received = set(agents)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ValueError(f"Missing rollout agents: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected rollout agents: {sorted(unexpected)}")


def _format_agent_list(agent_ids: list[str]) -> str:
    return ", ".join(agent_ids) if agent_ids else "none"
