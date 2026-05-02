from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from marlsim.core.environment import StepResult
from marlsim.core.state import Position


@dataclass(frozen=True)
class RunSummary:
    """Final metrics for one non-learning scenario run."""

    scenario_name: str
    total_steps: int
    succeeded: bool
    final_positions: dict[str, Position]
    reached_goals: frozenset[str]
    blocked_moves: int
    blocked_reasons: dict[str, int]


class ExperimentLogger:
    """Collect simple run metrics from environment StepResult objects."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self._blocked_reasons: Counter[str] = Counter()
        self._last_result: StepResult | None = None

    def record_step(self, result: StepResult) -> None:
        self._last_result = result

        for event in result.events:
            if event.blocked_reason is not None:
                self._blocked_reasons[event.blocked_reason] += 1

    def summary(self) -> RunSummary:
        if self._last_result is None:
            raise ValueError("Cannot summarize a run before any steps are recorded.")

        return RunSummary(
            scenario_name=self.scenario_name,
            total_steps=self._last_result.step_count,
            succeeded=len(self._last_result.reached_goals) == len(self._last_result.agents),
            final_positions={
                agent_id: agent.position
                for agent_id, agent in sorted(self._last_result.agents.items())
            },
            reached_goals=self._last_result.reached_goals,
            blocked_moves=sum(self._blocked_reasons.values()),
            blocked_reasons=dict(sorted(self._blocked_reasons.items())),
        )


def format_run_summary(summary: RunSummary) -> str:
    """Format run metrics as clean console text."""

    result = "success" if summary.succeeded else "failed"
    reached_goals = ", ".join(sorted(summary.reached_goals)) or "none"
    blocked_reasons = _format_blocked_reasons(summary.blocked_reasons)

    lines = [
        "Run summary",
        f"- scenario: {summary.scenario_name}",
        f"- result: {result}",
        f"- total steps: {summary.total_steps}",
        f"- reached goals: {reached_goals}",
        f"- blocked moves: {summary.blocked_moves}",
        f"- blocked reasons: {blocked_reasons}",
        "Final positions:",
    ]
    for agent_id, position in summary.final_positions.items():
        lines.append(f"- {agent_id}: {_format_position(position)}")

    return "\n".join(lines)


def _format_blocked_reasons(blocked_reasons: dict[str, int]) -> str:
    if not blocked_reasons:
        return "none"

    return ", ".join(
        f"{reason}={count}" for reason, count in sorted(blocked_reasons.items())
    )


def _format_position(position: Position) -> str:
    return f"({position.x}, {position.y})"
