from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Position:
    """A grid coordinate using x as column and y as row."""

    x: int
    y: int

    def move(self, dx: int, dy: int) -> "Position":
        return Position(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class AgentState:
    """State owned by one simulated agent."""

    agent_id: str
    position: Position
    goal: Position | None = None

    @property
    def at_goal(self) -> bool:
        return self.goal is not None and self.position == self.goal
