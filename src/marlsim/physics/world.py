from __future__ import annotations

from collections.abc import Sequence


class PhysicsDependencyError(RuntimeError):
    """Raised when optional PyBullet support is not installed."""


def _load_pybullet():
    try:
        import pybullet
    except ImportError as exc:
        raise PhysicsDependencyError(
            "PyBullet is required for physics simulation. "
            "Install the optional physics dependencies with "
            "`python -m pip install -e .[physics]`."
        ) from exc
    return pybullet


class PhysicsWorld:
    """Small PyBullet world helper for the Phase 7 setup spike."""

    def __init__(
        self,
        *,
        mode: str = "DIRECT",
        time_step: float = 1.0 / 240.0,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
    ) -> None:
        self.mode = mode
        self.time_step = time_step
        self.gravity = tuple(gravity)
        self.client_id: int | None = None
        self.plane_id: int | None = None

    @property
    def is_connected(self) -> bool:
        return self.client_id is not None

    def connect(self) -> int:
        if self.client_id is not None:
            return self.client_id

        pybullet = _load_pybullet()
        connection_mode = getattr(pybullet, self.mode.upper(), None)
        if connection_mode is None:
            raise ValueError(f"Unsupported PyBullet connection mode: {self.mode!r}")

        client_id = pybullet.connect(connection_mode)
        if client_id < 0:
            raise RuntimeError(f"Failed to connect to PyBullet in {self.mode!r} mode.")

        pybullet.setTimeStep(self.time_step, physicsClientId=client_id)
        pybullet.setGravity(*self.gravity, physicsClientId=client_id)
        self.client_id = client_id
        return client_id

    def create_plane(self) -> int:
        client_id = self._require_client()
        if self.plane_id is not None:
            return self.plane_id

        pybullet = _load_pybullet()
        collision_shape = pybullet.createCollisionShape(
            pybullet.GEOM_PLANE,
            physicsClientId=client_id,
        )
        self.plane_id = pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            physicsClientId=client_id,
        )
        return self.plane_id

    def create_robot(
        self,
        *,
        position: Sequence[float] = (0.0, 0.0, 0.15),
        half_extents: Sequence[float] = (0.2, 0.15, 0.1),
        mass: float = 1.0,
    ):
        from marlsim.physics.robot import SimpleRobot

        client_id = self._require_client()
        return SimpleRobot.create_box(
            client_id=client_id,
            position=position,
            half_extents=half_extents,
            mass=mass,
        )

    def create_static_obstacle(
        self,
        *,
        position: Sequence[float],
        half_extents: Sequence[float] = (0.25, 0.25, 0.25),
    ):
        from marlsim.physics.scenarios import create_static_obstacle

        client_id = self._require_client()
        return create_static_obstacle(
            client_id=client_id,
            position=position,
            half_extents=half_extents,
        )

    def create_goal_marker(
        self,
        *,
        position: Sequence[float],
        radius: float = 0.25,
    ):
        from marlsim.physics.scenarios import create_goal_marker

        client_id = self._require_client()
        return create_goal_marker(
            client_id=client_id,
            position=position,
            radius=radius,
        )

    def robot_reached_goal(
        self,
        robot,
        goal,
        *,
        threshold: float | None = None,
    ) -> bool:
        from marlsim.physics.scenarios import goal_reached

        self._require_client()
        return goal_reached(robot, goal, threshold=threshold)

    def move_robot(self, robot, displacement: Sequence[float], *, step: bool = True) -> None:
        self._require_client()
        robot.move_by(displacement)
        if step:
            self.step()

    def bodies_in_contact(self, body_a: int, body_b: int) -> bool:
        client_id = self._require_client()
        pybullet = _load_pybullet()
        contacts = pybullet.getContactPoints(
            bodyA=body_a,
            bodyB=body_b,
            physicsClientId=client_id,
        )
        return len(contacts) > 0

    def robots_in_contact(self, robot_a, robot_b) -> bool:
        return self.bodies_in_contact(robot_a.body_id, robot_b.body_id)

    def robot_touching_obstacle(self, robot, obstacle) -> bool:
        return self.bodies_in_contact(robot.body_id, obstacle.body_id)

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative.")

        client_id = self._require_client()
        pybullet = _load_pybullet()
        for _ in range(steps):
            pybullet.stepSimulation(physicsClientId=client_id)

    def disconnect(self) -> None:
        if self.client_id is None:
            return

        pybullet = _load_pybullet()
        pybullet.disconnect(physicsClientId=self.client_id)
        self.client_id = None
        self.plane_id = None

    def _require_client(self) -> int:
        if self.client_id is None:
            raise RuntimeError("PhysicsWorld must be connected before use.")
        return self.client_id

    def __enter__(self) -> PhysicsWorld:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()
