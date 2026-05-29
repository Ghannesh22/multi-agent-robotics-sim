from __future__ import annotations

from marlsim.physics import PhysicsDependencyError, PhysicsWorld


def main() -> None:
    world = PhysicsWorld(mode="DIRECT")
    try:
        world.connect()
        plane_id = world.create_plane()
        world.step(steps=10)
    except PhysicsDependencyError as exc:
        print(exc)
        raise SystemExit(1) from exc
    finally:
        world.disconnect()

    print("PyBullet DIRECT setup succeeded.")
    print(f"Created plane body id: {plane_id}")
    print("Ran 10 simulation steps and disconnected cleanly.")


if __name__ == "__main__":
    main()
