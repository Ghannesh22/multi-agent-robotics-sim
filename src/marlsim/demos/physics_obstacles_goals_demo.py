from __future__ import annotations

from marlsim.physics import PhysicsDependencyError, PhysicsWorld


def main() -> None:
    world = PhysicsWorld(mode="DIRECT")
    try:
        world.connect()
        world.create_plane()
        robot = world.create_robot(position=(0.0, 0.0, 0.2))
        obstacle = world.create_static_obstacle(
            position=(0.75, 0.0, 0.25),
            half_extents=(0.2, 0.2, 0.25),
        )
        goal = world.create_goal_marker(position=(0.0, 0.0, 0.2), radius=0.35)
        world.step(steps=30)
        robot_position = robot.position
        reached_goal = world.robot_reached_goal(robot, goal)
    except PhysicsDependencyError as exc:
        print(exc)
        raise SystemExit(1) from exc
    finally:
        world.disconnect()

    print("PyBullet obstacles and goals demo succeeded.")
    print(f"Robot body id: {robot.body_id}")
    print(f"Obstacle body id: {obstacle.body_id}")
    print(f"Goal marker body id: {goal.body_id}")
    print(f"Robot position: {robot_position}")
    print(f"Goal position: {goal.position}")
    print(f"Goal reached: {reached_goal}")
    print("Disconnected cleanly.")


if __name__ == "__main__":
    main()
