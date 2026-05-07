from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import MARLRewardMode, MultiAgentRLEnv, QLearningAgent, train_independent_q_learning


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


def _make_agents(env: MultiAgentRLEnv) -> dict[str, QLearningAgent]:
    return {
        agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=index)
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }


def run_comparison() -> str:
    rows = []
    for mode in (MARLRewardMode.INDIVIDUAL, MARLRewardMode.SHARED):
        env = _make_env()
        metrics = train_independent_q_learning(
            env=env,
            agents=_make_agents(env),
            episodes=50,
            reward_mode=mode,
        )
        rows.append(
            (
                mode.value,
                f"{metrics.success_rate:.2f}",
                f"{metrics.timeout_frequency:.2f}",
                str(metrics.blocked_move_count),
                f"{metrics.average_reward:.2f}",
                f"{metrics.average_episode_length:.2f}",
            )
        )

    headers = ("reward mode", "success", "timeouts", "blocked", "avg reward", "avg steps")
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


def main() -> None:
    print("MARL Reward Mode Comparison")
    print("===========================")
    print("Two independent Q-learning agents; no deep RL or communication.")
    print()
    print(run_comparison())


if __name__ == "__main__":
    main()
