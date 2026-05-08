from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    CommunicationAwareQLearningAgent,
    CommunicatingMultiAgentRLEnv,
    MultiAgentRLEnv,
    QLearningAgent,
    format_communication_rollout,
    format_message_timeline,
    rollout_communication_q_learning_trace,
    rollout_independent_communication_trace,
    rollout_rule_based_communication_trace,
    summarize_communication_episode,
    train_communication_q_learning,
    train_independent_q_learning,
    visualize_conflict_predictions,
    visualize_waiting_behavior,
)


def main() -> None:
    print(build_demo_output())


def build_demo_output() -> str:
    independent_agents = _train_independent_agents()
    communication_agents = _train_communication_agents()

    independent_trajectory = rollout_independent_communication_trace(
        env=MultiAgentRLEnv(_make_env(), ("agent_1", "agent_2")),
        agents=independent_agents,
    )
    rule_based_trajectory = rollout_rule_based_communication_trace(
        env=CommunicatingMultiAgentRLEnv(_make_env(), ("agent_1", "agent_2")),
    )
    conflict_trajectory = rollout_rule_based_communication_trace(
        env=CommunicatingMultiAgentRLEnv(_make_conflict_env(), ("agent_1", "agent_2")),
        mode_name="rule-based communication conflict trace",
    )
    communication_q_trajectory = rollout_communication_q_learning_trace(
        env=CommunicatingMultiAgentRLEnv(_make_env(), ("agent_1", "agent_2")),
        agents=communication_agents,
    )

    return "\n\n".join(
        (
            "Communication Visualization Demo\n==============================",
            summarize_communication_episode("independent MARL", independent_trajectory),
            format_communication_rollout(independent_trajectory),
            summarize_communication_episode("rule-based communication", rule_based_trajectory),
            format_communication_rollout(rule_based_trajectory),
            format_message_timeline(rule_based_trajectory),
            visualize_waiting_behavior(rule_based_trajectory),
            visualize_conflict_predictions(rule_based_trajectory),
            summarize_communication_episode(
                "rule-based communication conflict trace",
                conflict_trajectory,
            ),
            format_communication_rollout(conflict_trajectory),
            format_message_timeline(conflict_trajectory),
            visualize_waiting_behavior(conflict_trajectory),
            visualize_conflict_predictions(conflict_trajectory),
            summarize_communication_episode(
                "communication-aware Q-learning",
                communication_q_trajectory,
            ),
            format_communication_rollout(communication_q_trajectory),
            format_message_timeline(communication_q_trajectory),
            visualize_waiting_behavior(communication_q_trajectory),
            visualize_conflict_predictions(communication_q_trajectory),
            (
                "Observation note: deterministic rule-based communication is easy to inspect. "
                "Communication-aware Q-learning can still wait unnecessarily or underperform "
                "when extra message features fragment the tabular state space."
            ),
        )
    )


def _train_independent_agents() -> dict[str, QLearningAgent]:
    env = MultiAgentRLEnv(_make_env(), ("agent_1", "agent_2"))
    agents = {
        agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=index)
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }
    train_independent_q_learning(env=env, agents=agents, episodes=50)
    return agents


def _train_communication_agents() -> dict[str, CommunicationAwareQLearningAgent]:
    env = CommunicatingMultiAgentRLEnv(_make_env(), ("agent_1", "agent_2"))
    agents = {
        agent_id: CommunicationAwareQLearningAgent(
            action_size=env.action_size,
            epsilon=0.2,
            seed=100 + index,
        )
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }
    train_communication_q_learning(env=env, agents=agents, episodes=50)
    return agents


def _make_env() -> GridWorldEnv:
    return GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=2,
            agents=(
                AgentState("agent_1", Position(0, 0), Position(2, 0)),
                AgentState("agent_2", Position(0, 1), Position(2, 1)),
            ),
            max_steps=5,
        )
    )


def _make_conflict_env() -> GridWorldEnv:
    return GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=3,
            agents=(
                AgentState("agent_1", Position(0, 1), Position(1, 1)),
                AgentState("agent_2", Position(2, 1), Position(1, 1)),
            ),
            max_steps=4,
        )
    )


if __name__ == "__main__":
    main()
