from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    """Discrete movement actions for a grid-world agent."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"

    @property
    def delta(self) -> tuple[int, int]:
        match self:
            case Action.UP:
                return (0, -1)
            case Action.DOWN:
                return (0, 1)
            case Action.LEFT:
                return (-1, 0)
            case Action.RIGHT:
                return (1, 0)
            case Action.STAY:
                return (0, 0)

    @classmethod
    def from_value(cls, value: "Action | str") -> "Action":
        if isinstance(value, Action):
            return value
        return cls(value.lower())
