# Phase 7.4: Multi-Robot Physics Scene

Phase 7.4 adds a minimal multi-robot PyBullet scene to the optional physics layer.

This phase keeps physics separate from the grid-world simulator. It does not modify `GridWorldEnv`, RL wrappers, MARL wrappers, communication logic, reward logic, or path planning.

## Multi-Robot Scene

The physics world can now create multiple `SimpleRobot` bodies in the same PyBullet client.

Each robot is still a simple dynamic box body. The robots are independent PyBullet bodies with their own body ids and pose readback.

The goal is only to verify that multiple simple robot bodies can coexist in one physics scene.

## Movement Simplification

Movement is intentionally deterministic and direct.

`SimpleRobot.move_by(...)` shifts the robot body by a small `(dx, dy, dz)` displacement using PyBullet pose reset APIs. `PhysicsWorld.move_robot(...)` wraps that helper and can step the simulation immediately after the move.

This is not a realistic controller. It does not model motors, wheels, forces, acceleration, velocity commands, or continuous-control learning.

The simplified movement helper exists only so early demos and tests can move bodies around before adding real robot control concepts.

## Contact Checking

Phase 7.4 adds simple contact helpers:

- `PhysicsWorld.robots_in_contact(robot_a, robot_b)`
- `PhysicsWorld.robot_touching_obstacle(robot, obstacle)`
- `PhysicsWorld.bodies_in_contact(body_a, body_b)`

These helpers use PyBullet contact points and return booleans. They are intended for simple inspection, demos, and tests.

## How To Run The Demo

Use the verified Conda physics environment:

```powershell
conda activate marlsim-physics
python -m marlsim.demos.physics_multi_robot_demo
```

Expected output:

```text
PyBullet multi-robot demo succeeded.
Robot A body id: 1
Robot B body id: 2
Robot A initial position: (...)
Robot B initial position: (...)
Robot A final position: (...)
Robot B final position: (...)
Robots contacted: False
Disconnected cleanly.
```

Exact body ids, floating-point positions, and contact status may vary depending on the movement values.

## Limitations

Phase 7.4 does not include:

- RL integration.
- MARL integration.
- Communication integration.
- Sensors.
- Cameras.
- Path planning in physics.
- Complex controllers.
- Wheel or joint models.
- ROS.
- Gazebo.
- Deep RL.
- Changes to `GridWorldEnv`.

The multi-robot scene is still a physics primitive, not a learning environment.

## Why RL And Control Integration Remain Out Of Scope

Direct displacement is useful for setup and inspection, but it is not a robotics control interface.

Before RL or MARL can be connected safely, the project needs clearer physics task semantics: movement commands, collision handling, episode state, reward design, observations, and reset behavior.

Those are intentionally deferred until the physics building blocks are stable.

## Next Step

The next milestone is Phase 7.5: physics-based demo and logging.

Phase 7.5 should run a simple physics demo and log positions, contacts, and success without connecting physics to RL or MARL yet.
