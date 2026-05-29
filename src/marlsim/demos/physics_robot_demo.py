from __future__ import annotations

from marlsim.physics import PhysicsDependencyError, PhysicsWorld


def main() -> None:
    world = PhysicsWorld(mode="DIRECT")
    try:
        world.connect()
        world.create_plane()
        robot = world.create_robot(position=(0.0, 0.0, 0.2))
        initial_position = robot.position
        world.step(steps=30)
        final_position = robot.position
    except PhysicsDependencyError as exc:
        print(exc)
        raise SystemExit(1) from exc
    finally:
        world.disconnect()

    print("PyBullet single robot demo succeeded.")
    print(f"Robot body id: {robot.body_id}")
    print(f"Initial position: {initial_position}")
    print(f"Final position: {final_position}")
    print("Disconnected cleanly.")


if __name__ == "__main__":
    main()
