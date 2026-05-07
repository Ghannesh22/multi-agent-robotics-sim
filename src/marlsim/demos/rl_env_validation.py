from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import SingleAgentRLEnv


def _make_rl_env(*, max_steps: int = 5) -> SingleAgentRLEnv:
    env = GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=1,
            agents=(AgentState("agent_1", Position(0, 0), Position(2, 0)),),
            max_steps=max_steps,
        )
    )
    return SingleAgentRLEnv(env)


def _print_step(label: str, result: tuple[tuple[int, int, int, int], float, bool, dict[str, object]]) -> None:
    observation, reward, done, info = result
    print(label)
    print(f"  observation: {observation}")
    print(f"  reward: {reward}")
    print(f"  done: {done}")
    print(f"  info: {info}")
    print()


def run_validation() -> None:
    print("RL Environment Validation Demo")
    print("==============================")
    print("This is a scripted wrapper validation, not RL training.")
    print()

    env = _make_rl_env()
    print("Observation names:", env.observation_names)
    print("Action names:", env.action_names)
    print("Reward values:", env.reward_values)
    print()

    print("1. Reset output")
    print("  observation:", env.reset())
    print()

    print("2. Normal movement step")
    _print_step("  step(3) = right", env.step(3))

    blocked_env = _make_rl_env()
    blocked_env.reset()
    print("3. Blocked movement step")
    _print_step("  step(2) = left into out-of-bounds", blocked_env.step(2))

    goal_env = _make_rl_env()
    print("4. Goal-reaching episode")
    print("  reset:", goal_env.reset())
    _print_step("  step(3) = right", goal_env.step(3))
    _print_step("  step(3) = right into goal", goal_env.step(3))

    timeout_env = _make_rl_env(max_steps=1)
    print("5. Timeout episode")
    print("  reset:", timeout_env.reset())
    _print_step("  step(4) = stay until max_steps", timeout_env.step(4))


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
