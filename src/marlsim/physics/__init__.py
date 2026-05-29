"""Optional physics simulation helpers."""

from marlsim.physics.robot import SimpleRobot
from marlsim.physics.scenarios import GoalMarker, StaticObstacle, goal_reached
from marlsim.physics.world import PhysicsDependencyError, PhysicsWorld

__all__ = [
    "GoalMarker",
    "PhysicsDependencyError",
    "PhysicsWorld",
    "SimpleRobot",
    "StaticObstacle",
    "goal_reached",
]
