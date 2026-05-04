from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from marlsim.agents import (
    AgentPolicy,
    ConflictAwareWaitingAgent,
    LocalReplanningAgent,
    PriorityBasedCoordinationAgent,
    ShortestPathAgent,
)
from marlsim.core.environment import GridWorldEnv
from marlsim.core.state import Position
from marlsim.experiments.logging import ExperimentLogger, RunSummary
from marlsim.scenarios import get_scenario, scenario_names

PolicyFactory = Callable[[], AgentPolicy]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    factory: PolicyFactory


@dataclass(frozen=True)
class CoordinationComparisonResult:
    scenario_name: str
    policy_name: str
    summary: RunSummary


COORDINATION_POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec("baseline", ShortestPathAgent),
    PolicySpec("waiting", ConflictAwareWaitingAgent),
    PolicySpec("priority", PriorityBasedCoordinationAgent),
    PolicySpec("replanning", LocalReplanningAgent),
)


def run_coordination_comparison(
    scenarios: tuple[str, ...] | None = None,
    policies: tuple[PolicySpec, ...] = COORDINATION_POLICIES,
) -> tuple[CoordinationComparisonResult, ...]:
    """Run each coordination policy on each scenario and collect summaries."""

    scenario_list = scenarios or scenario_names()
    results: list[CoordinationComparisonResult] = []

    for scenario_name in scenario_list:
        for policy in policies:
            summary = run_policy_on_scenario(scenario_name, policy.factory)
            results.append(
                CoordinationComparisonResult(
                    scenario_name=scenario_name,
                    policy_name=policy.name,
                    summary=summary,
                )
            )

    return tuple(results)


def run_policy_on_scenario(
    scenario_name: str,
    policy_factory: PolicyFactory,
) -> RunSummary:
    """Run one policy type on one named scenario."""

    env = GridWorldEnv(get_scenario(scenario_name))
    logger = ExperimentLogger(scenario_name)
    policies = {agent_id: policy_factory() for agent_id in env.agents}

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
        logger.record_step(result)

        if result.terminated:
            return logger.summary()


def format_comparison_table(results: tuple[CoordinationComparisonResult, ...]) -> str:
    """Format coordination summaries as a readable markdown-style table."""

    headers = (
        "scenario",
        "policy",
        "result",
        "steps",
        "blocked",
        "blocked reasons",
        "reached goals",
        "final positions",
    )
    rows = [
        (
            result.scenario_name,
            result.policy_name,
            "success" if result.summary.succeeded else "failed",
            str(result.summary.total_steps),
            str(result.summary.blocked_moves),
            _format_blocked_reasons(result.summary.blocked_reasons),
            _format_reached_goals(result.summary.reached_goals),
            _format_final_positions(result.summary.final_positions),
        )
        for result in results
    ]
    widths = _column_widths(headers, rows)
    separator = tuple("-" * width for width in widths)

    lines = [
        _format_row(headers, widths),
        _format_row(separator, widths),
    ]
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def _column_widths(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> tuple[int, ...]:
    return tuple(
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    )


def _format_row(row: tuple[str, ...], widths: tuple[int, ...]) -> str:
    cells = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
    return " | ".join(cells)


def _format_blocked_reasons(blocked_reasons: dict[str, int]) -> str:
    if not blocked_reasons:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in blocked_reasons.items())


def _format_reached_goals(reached_goals: frozenset[str]) -> str:
    return ", ".join(sorted(reached_goals)) or "none"


def _format_final_positions(final_positions: dict[str, Position]) -> str:
    return ", ".join(
        f"{agent_id}=({position.x},{position.y})"
        for agent_id, position in sorted(final_positions.items())
    )
