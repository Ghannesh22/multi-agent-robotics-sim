from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import compare_communication_modes, format_communication_table


def main() -> None:
    print("Communication Evaluation Comparison")
    print("===================================")
    print("No communication vs rule-based communication vs communication-aware Q-learning.")
    print()
    print(run_comparison())


def run_comparison() -> str:
    return format_communication_table(compare_communication_modes(_make_env, episodes=50))


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
