from __future__ import annotations

import importlib.util
import unittest

from marlsim.physics import PhysicsWorld, SimpleRobot


class SimpleRobotTests(unittest.TestCase):
    def test_robot_stores_client_and_body_ids(self) -> None:
        robot = SimpleRobot(client_id=1, body_id=2)

        self.assertEqual(robot.client_id, 1)
        self.assertEqual(robot.body_id, 2)

    @unittest.skipIf(
        importlib.util.find_spec("pybullet") is None,
        "PyBullet optional dependency is not installed.",
    )
    def test_direct_world_creates_robot_and_reads_position(self) -> None:
        world = PhysicsWorld(mode="DIRECT")

        try:
            world.connect()
            world.create_plane()
            robot = world.create_robot(position=(0.0, 0.0, 0.2))
            world.step(steps=3)

            self.assertIsInstance(robot.body_id, int)
            self.assertEqual(len(robot.position), 3)
            self.assertEqual(len(robot.orientation), 4)
        finally:
            world.disconnect()
