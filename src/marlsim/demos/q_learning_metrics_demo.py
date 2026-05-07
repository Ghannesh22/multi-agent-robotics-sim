from __future__ import annotations

import io

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    QLearningAgent,
    SingleAgentRLEnv,
    export_training_metrics,
    reward_chart,
    success_rate_chart,
    summarize_training,
    train_q_learning,
)


def _make_training_env() -> SingleAgentRLEnv:
    env = GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=1,
            agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
            max_steps=5,
        )
    )
    return SingleAgentRLEnv(env)


def main() -> None:
    env = _make_training_env()
    agent = QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=7)
    metrics = train_q_learning(env=env, agent=agent, episodes=30)

    print("Q-Learning Metrics Demo")
    print("=======================")
    print("Dependency-free text reporting for tabular Q-learning.")
    print()
    print(summarize_training(metrics))
    print()
    print(reward_chart(metrics, width=24))
    print()
    print(success_rate_chart(metrics, window=5, width=24))
    print()

    csv_preview = io.StringIO()
    export_training_metrics(metrics, csv_preview)
    print("CSV preview")
    print("-----------")
    print("\n".join(csv_preview.getvalue().splitlines()[:6]))


if __name__ == "__main__":
    main()
