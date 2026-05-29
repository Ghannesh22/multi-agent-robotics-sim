from __future__ import annotations

from marlsim.physics import PhysicsDependencyError, PhysicsWorld


def main() -> None:
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
        world.step(steps=10)

        final_a = robot_a.position
        final_b = robot_b.position
        contacted = world.robots_in_contact(robot_a, robot_b)
    except PhysicsDependencyError as exc:
        print(exc)
        raise SystemExit(1) from exc
    finally:
        world.disconnect()

    print("PyBullet multi-robot demo succeeded.")
    print(f"Robot A body id: {robot_a.body_id}")
    print(f"Robot B body id: {robot_b.body_id}")
    print(f"Robot A initial position: {initial_a}")
    print(f"Robot B initial position: {initial_b}")
    print(f"Robot A final position: {final_a}")
    print(f"Robot B final position: {final_b}")
    print(f"Robots contacted: {contacted}")
    print("Disconnected cleanly.")


if __name__ == "__main__":
    main()
