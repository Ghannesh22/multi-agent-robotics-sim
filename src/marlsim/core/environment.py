from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from marlsim.core.actions import Action
from marlsim.core.state import AgentState, Position


class ConflictPolicy(str, Enum):
    """How same-cell movement conflicts are resolved."""

    BLOCK_ALL = "block_all"
    PRIORITY = "priority"


@dataclass(frozen=True)
class GridWorldConfig:
    width: int
    height: int
    obstacles: frozenset[Position] = field(default_factory=frozenset)
    agents: tuple[AgentState, ...] = field(default_factory=tuple)
    max_steps: int = 100
    conflict_policy: ConflictPolicy = ConflictPolicy.BLOCK_ALL


@dataclass(frozen=True)
class AgentStepEvent:
    agent_id: str
    action: Action
    start: Position
    requested: Position
    end: Position
    moved: bool
    blocked_reason: str | None = None


@dataclass(frozen=True)
class StepResult:
    agents: dict[str, AgentState]
    events: tuple[AgentStepEvent, ...]
    reached_goals: frozenset[str]
    step_count: int
    terminated: bool


class GridWorldEnv:
    """A deterministic simultaneous-move grid-world for multiple agents."""

    def __init__(self, config: GridWorldConfig):
        self.config = config
        self._validate_config(config)
        self._initial_agents = {agent.agent_id: agent for agent in config.agents}
        self.agents: dict[str, AgentState] = {}
        self.step_count = 0
        self.reset()

    def reset(self) -> dict[str, AgentState]:
        self.agents = dict(self._initial_agents)
        self.step_count = 0
        return self.snapshot()

    def snapshot(self) -> dict[str, AgentState]:
        return dict(self.agents)

    def step(self, actions: Mapping[str, Action | str]) -> StepResult:
        starts = {agent_id: agent.position for agent_id, agent in self.agents.items()}
        normalized_actions = {
            agent_id: Action.from_value(actions.get(agent_id, Action.STAY))
            for agent_id in self.agents
        }

        requested: dict[str, Position] = {}
        final: dict[str, Position] = {}
        blocked: dict[str, str | None] = {agent_id: None for agent_id in self.agents}

        for agent_id, action in normalized_actions.items():
            dx, dy = action.delta
            target = starts[agent_id].move(dx, dy)
            requested[agent_id] = target

            if target == starts[agent_id]:
                final[agent_id] = starts[agent_id]
            elif not self.in_bounds(target):
                final[agent_id] = starts[agent_id]
                blocked[agent_id] = "out_of_bounds"
            elif target in self.config.obstacles:
                final[agent_id] = starts[agent_id]
                blocked[agent_id] = "obstacle"
            else:
                final[agent_id] = target

        self._block_direct_swaps(starts, final, blocked)
        self._block_moves_into_stationary_agents(starts, final, blocked)
        self._resolve_same_cell_conflicts(final, blocked)
        self._block_moves_into_stationary_agents(starts, final, blocked)

        self.agents = {
            agent_id: AgentState(
                agent_id=agent_id,
                position=final[agent_id],
                goal=self.agents[agent_id].goal,
            )
            for agent_id in self.agents
        }
        self.step_count += 1

        events = tuple(
            AgentStepEvent(
                agent_id=agent_id,
                action=normalized_actions[agent_id],
                start=starts[agent_id],
                requested=requested[agent_id],
                end=final[agent_id],
                moved=starts[agent_id] != final[agent_id],
                blocked_reason=blocked[agent_id],
            )
            for agent_id in sorted(self.agents)
        )
        reached_goals = frozenset(
            agent_id for agent_id, agent in self.agents.items() if agent.at_goal
        )
        terminated = self.step_count >= self.config.max_steps or len(reached_goals) == len(
            self.agents
        )
        return StepResult(
            agents=self.snapshot(),
            events=events,
            reached_goals=reached_goals,
            step_count=self.step_count,
            terminated=terminated,
        )

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.config.width and 0 <= position.y < self.config.height

    def occupied_positions(self) -> dict[Position, str]:
        return {agent.position: agent_id for agent_id, agent in self.agents.items()}

    def _block_direct_swaps(
        self,
        starts: Mapping[str, Position],
        final: dict[str, Position],
        blocked: dict[str, str | None],
    ) -> None:
        agent_ids = sorted(starts)
        for index, left_id in enumerate(agent_ids):
            for right_id in agent_ids[index + 1 :]:
                left_swaps = final[left_id] == starts[right_id]
                right_swaps = final[right_id] == starts[left_id]
                if left_swaps and right_swaps:
                    final[left_id] = starts[left_id]
                    final[right_id] = starts[right_id]
                    blocked[left_id] = "swap_conflict"
                    blocked[right_id] = "swap_conflict"

    def _resolve_same_cell_conflicts(
        self, final: dict[str, Position], blocked: dict[str, str | None]
    ) -> None:
        by_target: dict[Position, list[str]] = defaultdict(list)
        for agent_id, target in final.items():
            by_target[target].append(agent_id)

        for target, agent_ids in by_target.items():
            if len(agent_ids) <= 1:
                continue

            if self.config.conflict_policy == ConflictPolicy.PRIORITY:
                winner = sorted(agent_ids)[0]
                losers = [agent_id for agent_id in agent_ids if agent_id != winner]
            else:
                losers = agent_ids

            for agent_id in losers:
                final[agent_id] = self.agents[agent_id].position
                blocked[agent_id] = "cell_conflict"

    def _block_moves_into_stationary_agents(
        self,
        starts: Mapping[str, Position],
        final: dict[str, Position],
        blocked: dict[str, str | None],
    ) -> None:
        changed = True
        while changed:
            changed = False
            stationary_positions = {
                start: agent_id
                for agent_id, start in starts.items()
                if final[agent_id] == start
            }

            for agent_id, target in list(final.items()):
                if target == starts[agent_id]:
                    continue

                stationary_agent_id = stationary_positions.get(target)
                if stationary_agent_id is not None and stationary_agent_id != agent_id:
                    final[agent_id] = starts[agent_id]
                    blocked[agent_id] = "occupied"
                    changed = True

            if changed:
                self._resolve_same_cell_conflicts(final, blocked)

    @staticmethod
    def _validate_config(config: GridWorldConfig) -> None:
        if config.width <= 0 or config.height <= 0:
            raise ValueError("Grid dimensions must be positive.")

        seen_agent_ids: set[str] = set()
        seen_positions: set[Position] = set()

        for obstacle in config.obstacles:
            if not (0 <= obstacle.x < config.width and 0 <= obstacle.y < config.height):
                raise ValueError(f"Obstacle outside grid: {obstacle}")

        for agent in config.agents:
            if agent.agent_id in seen_agent_ids:
                raise ValueError(f"Duplicate agent id: {agent.agent_id}")
            seen_agent_ids.add(agent.agent_id)

            if not (0 <= agent.position.x < config.width and 0 <= agent.position.y < config.height):
                raise ValueError(f"Agent outside grid: {agent}")
            if agent.position in config.obstacles:
                raise ValueError(f"Agent starts on obstacle: {agent}")
            if agent.position in seen_positions:
                raise ValueError(f"Multiple agents start at {agent.position}")
            seen_positions.add(agent.position)

            if agent.goal is not None:
                if not (0 <= agent.goal.x < config.width and 0 <= agent.goal.y < config.height):
                    raise ValueError(f"Goal outside grid: {agent}")
                if agent.goal in config.obstacles:
                    raise ValueError(f"Goal on obstacle: {agent}")
