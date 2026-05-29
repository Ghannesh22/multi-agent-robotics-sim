from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import dist

from marlsim.physics.robot import SimpleRobot, Vector3
from marlsim.physics.world import _load_pybullet


@dataclass(frozen=True)
class StaticObstacle:
    """Static box obstacle in a PyBullet world."""

    client_id: int
    body_id: int
    position: Vector3
    half_extents: Vector3


@dataclass(frozen=True)
class GoalMarker:
    """Visual goal marker with radius-based reach detection."""

    client_id: int
    body_id: int
    position: Vector3
    radius: float


def create_static_obstacle(
    *,
    client_id: int,
    position: Sequence[float],
    half_extents: Sequence[float] = (0.25, 0.25, 0.25),
) -> StaticObstacle:
    pybullet = _load_pybullet()
    collision_shape = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=tuple(half_extents),
        physicsClientId=client_id,
    )
    visual_shape = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=tuple(half_extents),
        rgbaColor=(0.8, 0.2, 0.2, 1.0),
        physicsClientId=client_id,
    )
    body_id = pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=tuple(position),
        physicsClientId=client_id,
    )
    return StaticObstacle(
        client_id=client_id,
        body_id=body_id,
        position=tuple(position),
        half_extents=tuple(half_extents),
    )


def create_goal_marker(
    *,
    client_id: int,
    position: Sequence[float],
    radius: float = 0.25,
) -> GoalMarker:
    pybullet = _load_pybullet()
    visual_shape = pybullet.createVisualShape(
        pybullet.GEOM_SPHERE,
        radius=radius,
        rgbaColor=(0.1, 0.8, 0.2, 0.7),
        physicsClientId=client_id,
    )
    body_id = pybullet.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=tuple(position),
        physicsClientId=client_id,
    )
    return GoalMarker(
        client_id=client_id,
        body_id=body_id,
        position=tuple(position),
        radius=radius,
    )


def goal_reached(
    robot: SimpleRobot,
    goal: GoalMarker,
    *,
    threshold: float | None = None,
) -> bool:
    reach_radius = goal.radius if threshold is None else threshold
    return dist(robot.position, goal.position) <= reach_radius
