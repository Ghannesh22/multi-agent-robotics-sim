from __future__ import annotations

from collections import deque

from marlsim.core.actions import Action
from marlsim.core.state import Position


def shortest_path(
    start: Position,
    goal: Position,
    obstacles: frozenset[Position],
    width: int,
    height: int,
    blocked: frozenset[Position] = frozenset(),
) -> tuple[Position, ...] | None:
    """Return the shortest grid path from start to goal, including both endpoints."""

    if start == goal:
        return (start,)

    frontier: deque[Position] = deque([start])
    came_from: dict[Position, Position | None] = {start: None}

    while frontier:
        current = frontier.popleft()

        for action in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
            dx, dy = action.delta
            neighbor = current.move(dx, dy)

            if neighbor in came_from:
                continue
            if not (0 <= neighbor.x < width and 0 <= neighbor.y < height):
                continue
            if neighbor in obstacles or neighbor in blocked:
                continue

            came_from[neighbor] = current
            if neighbor == goal:
                return _reconstruct_path(came_from, goal)

            frontier.append(neighbor)

    return None


def _reconstruct_path(
    came_from: dict[Position, Position | None], goal: Position
) -> tuple[Position, ...]:
    path = [goal]
    current = goal

    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return tuple(path)
