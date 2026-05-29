# Phase 7.2: Single Robot Body

Phase 7.2 adds the first minimal robot body to the optional PyBullet physics path.

This milestone keeps physics isolated from the existing grid-world, RL, MARL, and communication systems. It does not change `GridWorldEnv`.

## Robot Body

The first robot is intentionally simple: one dynamic box body in PyBullet.

The `SimpleRobot` helper stores:

- `client_id`: the PyBullet physics client.
- `body_id`: the PyBullet rigid body id.

It exposes:

- `position`: the body's current `(x, y, z)` position.
- `orientation`: the body's current quaternion orientation.
- `reset_pose(...)`: a small helper for resetting the body pose.
- `remove()`: a small helper for removing the body from the simulation.

The body is not a realistic robot model yet. It has no wheels, joints, sensors, motors, controller, URDF, or learning policy.

## Why It Is Simple

The goal of Phase 7.2 is to verify the next smallest physics concept after Phase 7.1:

```text
world + plane + one robot body + simulation steps + pose readback
```

A box body is enough to confirm that the project can create a physical object, step the simulation, and read its pose. More realistic robot modeling belongs in later phases after the physics structure is stable.

## How To Run The Demo

Use the verified Conda physics environment:

```powershell
conda activate marlsim-physics
python -m marlsim.demos.physics_robot_demo
```

Expected output:

```text
PyBullet single robot demo succeeded.
Robot body id: 1
Initial position: (...)
Final position: (...)
Disconnected cleanly.
```

Exact body ids and floating-point positions may vary.

## What Is Intentionally Not Included Yet

Phase 7.2 does not include:

- Multiple robots.
- Obstacles.
- Goals.
- Sensors.
- Cameras.
- Robot learning.
- RL integration.
- MARL integration.
- Communication integration.
- Continuous-control learning.
- ROS.
- Gazebo.
- Deep RL.
- Changes to `GridWorldEnv`.

Those topics remain deferred to later phases.

## Next Step

The next milestone is Phase 7.3: obstacles and goals.

Phase 7.3 should add static obstacles, simple goal markers, and basic goal-reaching detection while keeping the physics simulator separate from the grid-world system.
