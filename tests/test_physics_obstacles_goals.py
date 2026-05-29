from __future__ import annotations

import importlib.util
import unittest

from marlsim.physics import GoalMarker, PhysicsWorld, goal_reached


class FakeRobot:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self.position = position


class PhysicsObstaclesGoalsTests(unittest.TestCase):
    def test_goal_reached_uses_distance_threshold(self) -> None:
        robot = FakeRobot(position=(0.3, 0.0, 0.0))
        goal = GoalMarker(
            client_id=1,
            body_id=3,
            position=(0.0, 0.0, 0.0),
            radius=0.5,
        )

        self.assertTrue(goal_reached(robot, goal))
        self.assertFalse(goal_reached(robot, goal, threshold=0.1))

    @unittest.skipIf(
        importlib.util.find_spec("pybullet") is None,
        "PyBullet optional dependency is not installed.",
    )
    def test_direct_world_creates_obstacle_goal_and_checks_reached(self) -> None:
        world = PhysicsWorld(mode="DIRECT")

        try:
            world.connect()
            world.create_plane()
            robot = world.create_robot(position=(0.0, 0.0, 0.2))
            obstacle = world.create_static_obstacle(position=(1.0, 0.0, 0.25))
            goal = world.create_goal_marker(position=(0.0, 0.0, 0.2), radius=0.5)

            self.assertIsInstance(obstacle.body_id, int)
            self.assertIsInstance(goal.body_id, int)
            self.assertTrue(world.robot_reached_goal(robot, goal))
            far_goal = world.create_goal_marker(position=(2.0, 0.0, 0.2), radius=0.2)
            self.assertFalse(world.robot_reached_goal(robot, far_goal))
        finally:
            world.disconnect()
