from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    CommunicationAwareQLearningAgent,
    CommunicatingMultiAgentRLEnv,
    MultiAgentRLEnv,
    QLearningAgent,
    train_communication_q_learning,
    train_independent_q_learning,
)


def main() -> None:
    print("Communication-Aware Q-Learning Comparison")
    print("=========================================")
    print("Independent tabular Q-learning vs message-augmented tabular Q-learning.")
    print()
    print(run_comparison())


def run_comparison() -> str:
    baseline_env = MultiAgentRLEnv(_make_env(), ("agent_1", "agent_2"))
    communication_env = CommunicatingMultiAgentRLEnv(_make_env(), ("agent_1", "agent_2"))

    baseline_metrics = train_independent_q_learning(
        env=baseline_env,
        agents={
            agent_id: QLearningAgent(action_size=baseline_env.action_size, epsilon=0.2, seed=index)
            for index, agent_id in enumerate(baseline_env.controlled_agent_ids, start=1)
        },
        episodes=50,
    )
    communication_metrics = train_communication_q_learning(
        env=communication_env,
        agents={
            agent_id: CommunicationAwareQLearningAgent(
                action_size=communication_env.action_size,
                epsilon=0.2,
                seed=index,
            )
            for index, agent_id in enumerate(communication_env.controlled_agent_ids, start=1)
        },
        episodes=50,
    )

    rows = [
        (
            "independent",
            f"{baseline_metrics.success_rate:.2f}",
            f"{baseline_metrics.timeout_frequency:.2f}",
            str(baseline_metrics.blocked_move_count),
            f"{baseline_metrics.average_reward:.2f}",
            f"{baseline_metrics.average_episode_length:.2f}",
        ),
        (
            "communication",
            f"{communication_metrics.success_rate:.2f}",
            f"{communication_metrics.timeout_frequency:.2f}",
            str(communication_metrics.blocked_move_count),
            f"{communication_metrics.average_reward:.2f}",
            f"{communication_metrics.average_episode_length:.2f}",
        ),
    ]
    headers = ("training", "success", "timeouts", "blocked", "avg reward", "avg steps")
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


if __name__ == "__main__":
    main()
