from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import QLearningAgent, SingleAgentRLEnv, train_q_learning


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
    episodes = 50

    metrics = train_q_learning(env=env, agent=agent, episodes=episodes)
    start_state = (0, 0, 2, 0)

    print("Q-Learning Demo")
    print("===============")
    print("This is simple tabular Q-learning, not deep RL.")
    print()
    print(f"Episodes: {metrics.total_episodes}")
    print(f"Successes: {metrics.success_count}")
    print(f"Failures: {metrics.failure_count}")
    print(f"Timeouts: {metrics.timeout_count}")
    print(f"Average reward: {metrics.average_reward:.2f}")
    print(f"Average episode length: {metrics.average_episode_length:.2f}")
    print()
    print(f"Learned Q-values for start state {start_state}:")
    for action_id, action_name in enumerate(env.action_names):
        print(f"  {action_id} ({action_name}): {agent.get_q_value(start_state, action_id):.3f}")


if __name__ == "__main__":
    main()
