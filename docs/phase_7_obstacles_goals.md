# Phase 7.3: Physics Obstacles And Goals

Phase 7.3 adds simple static obstacles, visual goal markers, and radius-based goal-reaching checks to the optional PyBullet physics layer.

This remains separate from the grid-world simulator. It does not modify `GridWorldEnv`, rewards, RL wrappers, MARL wrappers, communication logic, or planning logic.

## Obstacle Representation

An obstacle is represented as a static PyBullet box body.

The `StaticObstacle` helper stores:

- `client_id`: the PyBullet physics client.
- `body_id`: the PyBullet body id.
- `position`: the obstacle center position.
- `half_extents`: the box half-size in each axis.

Obstacles have collision and visual shapes. They have zero mass, so they remain fixed in the world.

## Goal Representation

A goal is represented as a visual PyBullet sphere marker.

The `GoalMarker` helper stores:

- `client_id`: the PyBullet physics client.
- `body_id`: the PyBullet body id.
- `position`: the goal center position.
- `radius`: the goal reach radius.

The marker is visual-only for now. It is intended to make the target location explicit without adding sensors, rewards, or control logic.

## Goal-Reaching Detection

Goal reaching is checked with a simple distance threshold:

```text
distance(robot.position, goal.position) <= goal.radius
```

A custom threshold can be supplied when a test or demo needs a stricter or looser check.

This is intentionally simpler than a full robotics task definition. It is enough to verify that the physics layer can represent a target and detect when a robot body is close to it.

## How To Run The Demo

Use the verified Conda physics environment:

```powershell
conda activate marlsim-physics
python -m marlsim.demos.physics_obstacles_goals_demo
```

Expected output:

```text
PyBullet obstacles and goals demo succeeded.
Robot body id: 1
Obstacle body id: 2
Goal marker body id: 3
Robot position: (...)
Goal position: (...)
Goal reached: True
Disconnected cleanly.
```

Exact body ids and floating-point positions may vary.

## Limitations

Phase 7.3 does not include:

- Path planning in the physics world.
- Movement commands toward goals.
- Collision-aware robot control.
- Multiple robots.
- Sensors.
- Cameras.
- RL integration.
- MARL integration.
- Communication integration.
- ROS.
- Gazebo.
- Deep RL.
- Changes to `GridWorldEnv`.

The obstacle and goal helpers are building blocks only. They do not yet define a physics task environment or learning interface.

## Why This Is Not RL Or Control Integration

The robot is not learning, planning, or controlling itself in this phase. The demo creates bodies, steps the simulation, and checks distance to a goal marker.

RL and control integration should wait until the physics scene primitives are stable and easy to inspect.

## Next Step

The next milestone is Phase 7.4: multi-robot physics scene.

Phase 7.4 should add multiple simple robot bodies, collision checks between bodies, and simple simultaneous movement while keeping the physics layer isolated from the grid-world system.
