from __future__ import annotations

from marlsim.core.environment import ConflictPolicy, GridWorldConfig
from marlsim.core.state import AgentState, Position


def two_agent_obstacle_course() -> GridWorldConfig:
    """Small deterministic scenario for the first project milestone."""

    return GridWorldConfig(
        width=7,
        height=5,
        obstacles=frozenset(
            {
                Position(3, 1),
                Position(3, 2),
                Position(3, 3),
            }
        ),
        agents=(
            AgentState("agent_1", Position(0, 0), Position(6, 0)),
            AgentState("agent_2", Position(0, 4), Position(6, 4)),
        ),
        max_steps=30,
        conflict_policy=ConflictPolicy.BLOCK_ALL,
    )
