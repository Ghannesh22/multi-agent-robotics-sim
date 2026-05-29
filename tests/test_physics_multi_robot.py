from __future__ import annotations

import importlib.util
import unittest

from marlsim.physics import PhysicsWorld


@unittest.skipIf(
    importlib.util.find_spec("pybullet") is None,
    "PyBullet optional dependency is not installed.",
)
class PhysicsMultiRobotTests(unittest.TestCase):
    def test_two_robots_can_move_and_report_contact_status(self) -> None:
        world = PhysicsWorld(mode="DIRECT")

        try:
            world.connect()
            world.create_plane()
            robot_a = world.create_robot(position=(-0.5, 0.0, 0.2))
            robot_b = world.create_robot(position=(0.5, 0.0, 0.2))

            initial_a = robot_a.position
            initial_b = robot_b.position

            world.move_robot(robot_a, (0.25, 0.0, 0.0), step=False)
            world.move_robot(robot_b, (-0.25, 0.0, 0.0), step=False)
            world.step(steps=2)

            self.assertNotEqual(robot_a.position, initial_a)
            self.assertNotEqual(robot_b.position, initial_b)
            self.assertIsInstance(world.robots_in_contact(robot_a, robot_b), bool)
        finally:
            world.disconnect()

    def test_robot_obstacle_contact_helper_returns_bool(self) -> None:
        world = PhysicsWorld(mode="DIRECT")

        try:
            world.connect()
            world.create_plane()
            robot = world.create_robot(position=(0.0, 0.0, 0.2))
            obstacle = world.create_static_obstacle(
                position=(0.0, 0.0, 0.2),
                half_extents=(0.2, 0.2, 0.2),
            )
            world.step(steps=2)

            self.assertIsInstance(world.robot_touching_obstacle(robot, obstacle), bool)
        finally:
            world.disconnect()
