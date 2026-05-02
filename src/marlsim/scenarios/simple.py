from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from marlsim.core.environment import ConflictPolicy, GridWorldConfig
from marlsim.core.state import AgentState, Position


@dataclass(frozen=True)
class AgentScenario:
    """Beginner-friendly input for one agent in a scenario."""

    agent_id: str
    start: tuple[int, int]
    goal: tuple[int, int] | None = None


@dataclass(frozen=True)
class Scenario:
    """Reusable grid-world scenario definition."""

    name: str
    width: int
    height: int
    agents: Sequence[AgentScenario]
    obstacles: Iterable[tuple[int, int]] = ()
    max_steps: int = 100
    conflict_policy: ConflictPolicy = ConflictPolicy.BLOCK_ALL

    def build(self) -> GridWorldConfig:
        return make_grid_world_config(
            width=self.width,
            height=self.height,
            agents=self.agents,
            obstacles=self.obstacles,
            max_steps=self.max_steps,
            conflict_policy=self.conflict_policy,
        )


def make_grid_world_config(
    width: int,
    height: int,
    agents: Sequence[AgentScenario],
    obstacles: Iterable[tuple[int, int]] = (),
    max_steps: int = 100,
    conflict_policy: ConflictPolicy = ConflictPolicy.BLOCK_ALL,
) -> GridWorldConfig:
    """Create a GridWorldConfig from simple tuples."""

    return GridWorldConfig(
        width=width,
        height=height,
        obstacles=frozenset(Position(x, y) for x, y in obstacles),
        agents=tuple(
            AgentState(
                agent_id=agent.agent_id,
                position=Position(*agent.start),
                goal=Position(*agent.goal) if agent.goal is not None else None,
            )
            for agent in agents
        ),
        max_steps=max_steps,
        conflict_policy=conflict_policy,
    )


def two_agent_obstacle_course() -> GridWorldConfig:
    """Small deterministic scenario for the first project milestone."""

    return Scenario(
        name="simple",
        width=7,
        height=5,
        obstacles=(
            (3, 1),
            (3, 2),
            (3, 3),
        ),
        agents=(
            AgentScenario("agent_1", (0, 0), (6, 0)),
            AgentScenario("agent_2", (0, 4), (6, 4)),
        ),
        max_steps=30,
        conflict_policy=ConflictPolicy.BLOCK_ALL,
    ).build()


def narrow_corridor() -> GridWorldConfig:
    """Two agents move through separate narrow lanes divided by obstacles."""

    return Scenario(
        name="narrow_corridor",
        width=7,
        height=5,
        obstacles=((3, 1), (3, 2), (3, 3)),
        agents=(
            AgentScenario("agent_1", (0, 2), (6, 2)),
            AgentScenario("agent_2", (6, 4), (0, 4)),
        ),
        max_steps=30,
        conflict_policy=ConflictPolicy.BLOCK_ALL,
    ).build()


def crossing_paths() -> GridWorldConfig:
    """Agents travel across an open intersection from different directions."""

    return Scenario(
        name="crossing_paths",
        width=5,
        height=5,
        obstacles=(),
        agents=(
            AgentScenario("agent_1", (0, 2), (4, 2)),
            AgentScenario("agent_2", (2, 0), (2, 4)),
        ),
        max_steps=20,
        conflict_policy=ConflictPolicy.PRIORITY,
    ).build()


ScenarioFactory = Callable[[], GridWorldConfig]

SCENARIOS: dict[str, ScenarioFactory] = {
    "simple": two_agent_obstacle_course,
    "narrow_corridor": narrow_corridor,
    "crossing_paths": crossing_paths,
}


def scenario_names() -> tuple[str, ...]:
    """Return available scenario names for demos and tests."""

    return tuple(SCENARIOS)


def get_scenario(name: str) -> GridWorldConfig:
    """Build a named scenario."""

    try:
        return SCENARIOS[name]()
    except KeyError as error:
        available = ", ".join(scenario_names())
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {available}") from error
