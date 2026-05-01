from __future__ import annotations

from marlsim.agents import ShortestPathAgent
from marlsim.core.environment import GridWorldEnv
from marlsim.scenarios.simple import two_agent_obstacle_course
from marlsim.visualization.console_renderer import ConsoleRenderer


def main() -> None:
    env = GridWorldEnv(two_agent_obstacle_course())
    renderer = ConsoleRenderer()
    policies = {agent_id: ShortestPathAgent() for agent_id in env.agents}

    print("Initial state")
    print(renderer.render(env))
    print()

    while True:
        actions = {
            agent_id: policies[agent_id].choose_action(
                agent=agent,
                agents=env.agents,
                obstacles=env.config.obstacles,
                width=env.config.width,
                height=env.config.height,
            )
            for agent_id, agent in env.agents.items()
        }
        result = env.step(actions)

        print(f"Step {result.step_count}: " + ", ".join(f"{k}={v.value}" for k, v in actions.items()))
        print(renderer.render(env))
        print()

        if result.terminated:
            reached = ", ".join(sorted(result.reached_goals)) or "none"
            print(f"Terminated after {result.step_count} steps. Reached goals: {reached}")
            break


if __name__ == "__main__":
    main()
