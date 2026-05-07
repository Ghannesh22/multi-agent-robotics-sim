from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import MultiAgentRLEnv, QLearningAgent, summarize_marl_training, train_independent_q_learning


def _make_env() -> MultiAgentRLEnv:
    env = GridWorldEnv(
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
    return MultiAgentRLEnv(env, ("agent_1", "agent_2"))


def main() -> None:
    env = _make_env()
    agents = {
        agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=index)
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }
    metrics = train_independent_q_learning(env=env, agents=agents, episodes=50)

    print("Independent MARL Q-Learning Demo")
    print("=================================")
    print("Two agents train with separate Q-tables. This is not deep RL.")
    print()
    print(summarize_marl_training(metrics))
    print()
    for agent_id, rewards in metrics.per_agent_rewards.items():
        average_reward = sum(rewards) / len(rewards)
        print(f"{agent_id} average reward: {average_reward:.2f}")


if __name__ == "__main__":
    main()
