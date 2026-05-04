from __future__ import annotations

from typing import Mapping

from marlsim.agents.base import AgentPolicy
from marlsim.agents.planning import ShortestPathAgent
from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position
from marlsim.planning import shortest_path


class ConflictAwareWaitingAgent(AgentPolicy):
    """Wait when the next preferred move would create an obvious conflict."""

    def __init__(self, base_policy: AgentPolicy | None = None):
        self.base_policy = base_policy or ShortestPathAgent()

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        intended_action = self.base_policy.choose_action(
            agent=agent,
            agents=agents,
            obstacles=obstacles,
            width=width,
            height=height,
        )
        intended_target = _target_position(agent.position, intended_action)

        if intended_action == Action.STAY:
            return Action.STAY

        for other_id, other in agents.items():
            if other_id == agent.agent_id:
                continue

            other_action = self.base_policy.choose_action(
                agent=other,
                agents=agents,
                obstacles=obstacles,
                width=width,
                height=height,
            )
            other_target = _target_position(other.position, other_action)

            same_cell_conflict = intended_target == other_target
            direct_swap_conflict = (
                intended_target == other.position and other_target == agent.position
            )

            if same_cell_conflict or direct_swap_conflict:
                return Action.STAY

        return intended_action


class PriorityBasedCoordinationAgent(AgentPolicy):
    """Yield only when another conflicting agent has higher priority."""

    def __init__(
        self,
        base_policy: AgentPolicy | None = None,
        priority_order: tuple[str, ...] | None = None,
    ):
        self.base_policy = base_policy or ShortestPathAgent()
        self.priority_order = priority_order

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        intended_actions = {
            agent_id: self.base_policy.choose_action(
                agent=other,
                agents=agents,
                obstacles=obstacles,
                width=width,
                height=height,
            )
            for agent_id, other in agents.items()
        }
        intended_action = intended_actions[agent.agent_id]
        intended_target = _target_position(agent.position, intended_action)

        if intended_action == Action.STAY:
            return Action.STAY

        intended_targets = {
            agent_id: _target_position(agents[agent_id].position, action)
            for agent_id, action in intended_actions.items()
        }

        for other_id, other in agents.items():
            if other_id == agent.agent_id:
                continue

            other_target = intended_targets[other_id]
            same_cell_conflict = intended_target == other_target
            direct_swap_conflict = (
                intended_target == other.position
                and other_target == agent.position
            )

            if same_cell_conflict:
                winner = self._highest_priority((agent.agent_id, other_id), agents)
                if winner != agent.agent_id:
                    return Action.STAY

            if direct_swap_conflict:
                winner = self._highest_priority((agent.agent_id, other_id), agents)
                if winner != agent.agent_id:
                    return Action.STAY
                return Action.STAY

            if intended_target == other.position and other_target == other.position:
                return Action.STAY

        return intended_action

    def _highest_priority(
        self, agent_ids: tuple[str, ...], agents: Mapping[str, AgentState]
    ) -> str:
        return min(agent_ids, key=lambda agent_id: self._priority_index(agent_id, agents))

    def _priority_index(self, agent_id: str, agents: Mapping[str, AgentState]) -> int:
        ordered_agent_ids = self.priority_order or tuple(sorted(agents))
        if agent_id in ordered_agent_ids:
            return ordered_agent_ids.index(agent_id)
        return len(ordered_agent_ids) + tuple(sorted(agents)).index(agent_id)


class LocalReplanningAgent(PriorityBasedCoordinationAgent):
    """Try a short alternate BFS route when the preferred next move is unsafe."""

    def choose_action(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> Action:
        intended_actions = self._intended_actions(
            agents=agents,
            obstacles=obstacles,
            width=width,
            height=height,
        )
        intended_action = intended_actions[agent.agent_id]

        if intended_action == Action.STAY or agent.goal is None:
            return Action.STAY

        unsafe_cells = self._unsafe_cells_for(
            agent=agent,
            agents=agents,
            intended_actions=intended_actions,
        )
        intended_target = _target_position(agent.position, intended_action)

        if intended_target not in unsafe_cells:
            return intended_action

        intended_targets = {
            agent_id: _target_position(other.position, intended_actions[agent_id])
            for agent_id, other in agents.items()
        }
        likely_next_cells = {
            target
            for agent_id, target in intended_targets.items()
            if agent_id != agent.agent_id
        }
        replan_blocked_cells = unsafe_cells | likely_next_cells

        blocked = frozenset(
            replan_blocked_cells
            | {
                other.position
                for other_id, other in agents.items()
                if other_id != agent.agent_id
            }
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

        alternate_action = _action_from_step(agent.position, path[1])
        alternate_target = _target_position(agent.position, alternate_action)
        if alternate_target in replan_blocked_cells:
            return Action.STAY

        return alternate_action

    def _unsafe_cells_for(
        self,
        agent: AgentState,
        agents: Mapping[str, AgentState],
        intended_actions: Mapping[str, Action],
    ) -> set[Position]:
        intended_targets = {
            agent_id: _target_position(other.position, intended_actions[agent_id])
            for agent_id, other in agents.items()
        }
        agent_target = intended_targets[agent.agent_id]
        unsafe_cells: set[Position] = set()

        for other_id, other in agents.items():
            if other_id == agent.agent_id:
                continue

            other_target = intended_targets[other_id]
            same_cell_conflict = agent_target == other_target
            direct_swap_conflict = (
                agent_target == other.position and other_target == agent.position
            )

            if same_cell_conflict:
                winner = self._highest_priority((agent.agent_id, other_id), agents)
                if winner != agent.agent_id:
                    unsafe_cells.add(agent_target)

            if direct_swap_conflict:
                unsafe_cells.add(agent_target)

            if agent_target == other.position and other_target == other.position:
                unsafe_cells.add(agent_target)

        return unsafe_cells

    def _intended_actions(
        self,
        agents: Mapping[str, AgentState],
        obstacles: frozenset[Position],
        width: int,
        height: int,
    ) -> dict[str, Action]:
        return {
            agent_id: self.base_policy.choose_action(
                agent=agent,
                agents=agents,
                obstacles=obstacles,
                width=width,
                height=height,
            )
            for agent_id, agent in agents.items()
        }


def _target_position(position: Position, action: Action) -> Position:
    dx, dy = action.delta
    return position.move(dx, dy)


def _action_from_step(start: Position, end: Position) -> Action:
    dx = end.x - start.x
    dy = end.y - start.y

    for action in Action:
        if action.delta == (dx, dy):
            return action

    raise ValueError(f"Positions are not adjacent: {start} -> {end}")
