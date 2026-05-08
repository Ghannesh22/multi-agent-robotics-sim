from __future__ import annotations

from collections.abc import Mapping

from marlsim.agents import CommunicationAwarePriorityAgent
from marlsim.communication import CommunicationMessage
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position
from marlsim.rl.communicating_multi_agent_env import CommunicatingMultiAgentRLEnv
from marlsim.rl.marl_training import (
    MARLRewardMode,
    MARLTrainingMetrics,
    apply_reward_mode,
)
from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.single_agent_env import Observation


CommunicationObservation = tuple[int, int, int, int, int, int, int, int]
CommunicationFeatureNames = tuple[str, str, str, str]

COMMUNICATION_FEATURE_NAMES: CommunicationFeatureNames = (
    "other_agent_waiting",
    "other_agent_priority",
    "predicted_conflict",
    "intended_same_target",
)


class CommunicationAwareQLearningAgent(QLearningAgent):
    """Tabular Q-learning agent whose states may include communication features."""

    observation_names: tuple[str, ...] = (
        "agent_x",
        "agent_y",
        "goal_x",
        "goal_y",
        *COMMUNICATION_FEATURE_NAMES,
    )
    observation_size: int = len(observation_names)


def build_communication_observation(
    *,
    agent_id: str,
    observation: Observation,
    agents: Mapping[str, AgentState],
    messages: Mapping[str, CommunicationMessage],
) -> CommunicationObservation:
    """Append symbolic communication features to a base MARL observation."""

    other_messages = {
        sender_id: message
        for sender_id, message in messages.items()
        if sender_id != agent_id and sender_id in agents
    }
    own_message = messages.get(agent_id)
    own_agent = agents[agent_id]
    own_target = _message_next_position(own_agent, own_message)

    other_agent_waiting = int(any(message.waiting for message in other_messages.values()))
    other_agent_priority = min(
        (message.priority for message in other_messages.values()),
        default=-1,
    )
    predicted_conflict = int(
        any(
            _messages_predict_conflict(
                own_agent=own_agent,
                own_target=own_target,
                other_agent=agents[other_id],
                other_message=message,
            )
            for other_id, message in other_messages.items()
        )
    )
    intended_same_target = int(
        any(
            own_agent.goal is not None
            and message.target_position is not None
            and message.target_position == own_agent.goal
            for message in other_messages.values()
        )
    )

    return (
        *observation,
        other_agent_waiting,
        other_agent_priority,
        predicted_conflict,
        intended_same_target,
    )


def extend_observation_with_messages(
    observations: Mapping[str, Observation],
    *,
    agents: Mapping[str, AgentState],
    messages: Mapping[str, CommunicationMessage],
) -> dict[str, CommunicationObservation]:
    """Build communication-aware observations for all controlled agents."""

    return {
        agent_id: build_communication_observation(
            agent_id=agent_id,
            observation=observation,
            agents=agents,
            messages=messages,
        )
        for agent_id, observation in observations.items()
    }


def train_communication_q_learning(
    env: CommunicatingMultiAgentRLEnv,
    agents: Mapping[str, CommunicationAwareQLearningAgent],
    episodes: int,
    reward_mode: MARLRewardMode | str = MARLRewardMode.INDIVIDUAL,
) -> MARLTrainingMetrics:
    """Train independent tabular Q-learning agents with message-augmented states."""

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
            messages = _build_step_messages(env)
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
            next_messages = {} if done else _build_step_messages(env)
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


def _build_step_messages(
    env: CommunicatingMultiAgentRLEnv,
) -> dict[str, CommunicationMessage]:
    policy = CommunicationAwarePriorityAgent()
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


def _message_next_position(
    agent: AgentState,
    message: CommunicationMessage | None,
) -> Position:
    if message is None:
        return agent.position
    return _target_position(agent.position, message.intended_action)


def _messages_predict_conflict(
    *,
    own_agent: AgentState,
    own_target: Position,
    other_agent: AgentState,
    other_message: CommunicationMessage,
) -> bool:
    other_target = _message_next_position(other_agent, other_message)
    same_next_cell = own_target == other_target
    direct_swap = own_target == other_agent.position and other_target == own_agent.position
    moving_into_waiting_agent = own_target == other_agent.position and other_message.waiting
    return same_next_cell or direct_swap or moving_into_waiting_agent


def _target_position(position: Position, action: Action | None) -> Position:
    if action is None:
        return position
    dx, dy = action.delta
    return position.move(dx, dy)


def _validate_agents(
    env: CommunicatingMultiAgentRLEnv,
    agents: Mapping[str, CommunicationAwareQLearningAgent],
) -> None:
    expected = set(env.controlled_agent_ids)
    received = set(agents)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ValueError(f"Missing communication-aware Q-learning agents: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected communication-aware Q-learning agents: {sorted(unexpected)}")


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
