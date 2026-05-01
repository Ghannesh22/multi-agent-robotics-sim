from __future__ import annotations

from typing import Mapping

from marlsim.agents.base import AgentPolicy
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position


class GreedyGoalAgent(AgentPolicy):
    """Move along the axis that most reduces Manhattan distance to the goal."""

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        if agent.goal is None or agent.position == agent.goal:
            return Action.STAY

        candidates = self._candidate_actions(agent.position, agent.goal)
        occupied = {
            other.position
            for other_id, other in agents.items()
            if other_id != agent.agent_id
        }

        for action in candidates:
            dx, dy = action.delta
            target = agent.position.move(dx, dy)
            if (
                0 <= target.x < width
                and 0 <= target.y < height
                and target not in obstacles
                and target not in occupied
            ):
                return action

        return Action.STAY

    @staticmethod
    def _candidate_actions(position: Position, goal: Position) -> tuple[Action, ...]:
        dx = goal.x - position.x
        dy = goal.y - position.y

        horizontal = Action.RIGHT if dx > 0 else Action.LEFT
        vertical = Action.DOWN if dy > 0 else Action.UP

        if abs(dx) >= abs(dy):
            primary = horizontal if dx != 0 else vertical
            secondary = vertical if dy != 0 else horizontal
        else:
            primary = vertical if dy != 0 else horizontal
            secondary = horizontal if dx != 0 else vertical

        actions = [primary]
        if secondary != primary:
            actions.append(secondary)

        for fallback in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
            if fallback not in actions:
                actions.append(fallback)

        actions.append(Action.STAY)
        return tuple(actions)
