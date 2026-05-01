from __future__ import annotations

from typing import Mapping

from marlsim.agents.base import AgentPolicy
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position
from marlsim.planning import shortest_path


class ShortestPathAgent(AgentPolicy):
    """Choose the first step on a shortest path to the agent's goal."""

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

        blocked = frozenset(
            other.position
            for other_id, other in agents.items()
            if other_id != agent.agent_id
        )
        path = shortest_path(
            start=agent.position,
            goal=agent.goal,
            obstacles=obstacles,
            width=width,
            height=height,
            blocked=blocked,
        )

        if path is None or len(path) < 2:
            return Action.STAY

        return _action_from_step(agent.position, path[1])


def _action_from_step(start: Position, end: Position) -> Action:
    dx = end.x - start.x
    dy = end.y - start.y

    for action in Action:
        if action.delta == (dx, dy):
            return action

    raise ValueError(f"Positions are not adjacent: {start} -> {end}")
