from __future__ import annotations

import argparse

from marlsim.agents import ShortestPathAgent
from marlsim.core.environment import GridWorldEnv
from marlsim.experiments import ExperimentLogger, format_run_summary
from marlsim.scenarios import get_scenario, scenario_names
from marlsim.visualization.console_renderer import ConsoleRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid-world scenario demo.")
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default="simple",
        help="Scenario to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env = GridWorldEnv(get_scenario(args.scenario))
    renderer = ConsoleRenderer()
    experiment = ExperimentLogger(args.scenario)
    policies = {agent_id: ShortestPathAgent() for agent_id in env.agents}

    print(f"Scenario: {args.scenario}")
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
        experiment.record_step(result)

        print(renderer.render(env))
        action_summary = ", ".join(
            f"{agent_id}={action.value}" for agent_id, action in actions.items()
        )
        print(f"Actions: {action_summary}")
        print()

        if result.terminated:
            print(format_run_summary(experiment.summary()))
            break


if __name__ == "__main__":
    main()
