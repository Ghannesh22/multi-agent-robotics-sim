from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from marlsim.physics.world import _load_pybullet

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class SimpleRobot:
    """Minimal single-body robot for early PyBullet demos."""

    client_id: int
    body_id: int

    @classmethod
    def create_box(
        cls,
        *,
        client_id: int,
        position: Sequence[float] = (0.0, 0.0, 0.15),
        half_extents: Sequence[float] = (0.2, 0.15, 0.1),
        mass: float = 1.0,
    ) -> SimpleRobot:
        pybullet = _load_pybullet()
        collision_shape = pybullet.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=tuple(half_extents),
            physicsClientId=client_id,
        )
        visual_shape = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=tuple(half_extents),
            rgbaColor=(0.1, 0.4, 0.9, 1.0),
            physicsClientId=client_id,
        )
        body_id = pybullet.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=tuple(position),
            physicsClientId=client_id,
        )
        return cls(client_id=client_id, body_id=body_id)

    @property
    def position(self) -> Vector3:
        position, _orientation = self._pose()
        return position

    @property
    def orientation(self) -> Quaternion:
        _position, orientation = self._pose()
        return orientation

    def reset_pose(
        self,
        *,
        position: Sequence[float] = (0.0, 0.0, 0.15),
        orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        pybullet = _load_pybullet()
        pybullet.resetBasePositionAndOrientation(
            self.body_id,
            tuple(position),
            tuple(orientation),
            physicsClientId=self.client_id,
        )

    def move_by(self, displacement: Sequence[float]) -> None:
        current_position = self.position
        next_position = tuple(
            current + delta for current, delta in zip(current_position, displacement)
        )
        self.reset_pose(position=next_position, orientation=self.orientation)

    def remove(self) -> None:
        pybullet = _load_pybullet()
        pybullet.removeBody(self.body_id, physicsClientId=self.client_id)

    def _pose(self) -> tuple[Vector3, Quaternion]:
        pybullet = _load_pybullet()
        position, orientation = pybullet.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.client_id,
        )
        return tuple(position), tuple(orientation)
