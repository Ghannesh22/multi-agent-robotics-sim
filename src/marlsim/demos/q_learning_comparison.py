from __future__ import annotations

from collections.abc import Callable

from marlsim.agents import GreedyGoalAgent, RandomAgent, ShortestPathAgent
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    QLearningAgent,
    SingleAgentRLEnv,
    evaluate_policy,
    format_evaluation_table,
    train_q_learning,
)


def _make_env() -> SingleAgentRLEnv:
    env = GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=1,
            agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
            max_steps=5,
        )
    )
    return SingleAgentRLEnv(env)


def run_comparison() -> str:
    train_env = _make_env()
    q_agent = QLearningAgent(action_size=train_env.action_size, epsilon=0.2, seed=7)
    train_q_learning(env=train_env, agent=q_agent, episodes=50)

    policy_factories: list[tuple[str, Callable[[], object]]] = [
        ("random", lambda: RandomAgent(seed=7)),
        ("greedy", GreedyGoalAgent),
        ("shortest_path", ShortestPathAgent),
        ("q_learning", lambda: q_agent),
    ]
    results = [
        evaluate_policy(
            env=_make_env(),
            policy=policy_factory(),  # type: ignore[arg-type]
            episodes=20,
            policy_name=policy_name,
        )
        for policy_name, policy_factory in policy_factories
    ]
    return format_evaluation_table(results)


def main() -> None:
    print("Q-Learning Policy Comparison")
    print("============================")
    print("Evaluation uses fixed episodes and disables Q-learning exploration.")
    print()
    print(run_comparison())


if __name__ == "__main__":
    main()
