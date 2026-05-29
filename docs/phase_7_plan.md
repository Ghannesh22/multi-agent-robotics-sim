# Phase 7 Plan: Physics-Based Simulation Upgrade

## 1. Phase 7 Vision

Phase 7 introduces a lightweight physics-based simulation path for the project.

The goal is to move beyond purely grid-based motion while keeping the repository clean, approachable, and beginner-friendly. The existing grid-world simulator remains the stable foundation for coordination, reinforcement learning, communication experiments, tests, and demos. Physics simulation should be added beside that system, not by rewriting it.

Phase 7.0 is planning-only. It defines the scope, architecture, simulator choice, testing strategy, and out-of-scope boundaries before any PyBullet dependency or physics code is added.

## 2. Why Phase 7 Matters

The current grid-world is useful because it makes coordination easy to inspect. Agents occupy cells, choose discrete actions, and collide according to simple rules. That abstraction made Phases 1 through 6 easier to understand, test, and explain.

Real robots do not move on perfect grid cells. They move continuously through space, have orientation, accelerate and decelerate, collide with objects, and sense the world through imperfect measurements. A robot's state is closer to position, velocity, heading, body shape, and sensor readings than a single grid coordinate.

Physics simulation introduces concepts that are essential for a robotics bridge:

- Continuous positions instead of only integer grid cells.
- Orientation and heading.
- Velocity and movement over time.
- Physical collisions with obstacles and other robots.
- Robot bodies with size and shape.
- Sensors or simplified sensor-like observations.

This phase matters because it connects the project from abstract multi-agent coordination toward real robotics concepts while avoiding the complexity of full robotics middleware.

## 3. Recommended Simulator

Phase 7 should start with PyBullet.

PyBullet is the recommended first simulator because:

- It is Python-friendly.
- It works on Windows.
- It is lightweight compared with Gazebo and ROS-based simulation stacks.
- It is suitable for simple robot bodies, obstacles, collisions, and visual demos.
- It can be introduced as an optional dependency instead of a required dependency for the whole project.

Gazebo, ROS, MuJoCo, Isaac Sim, Unity, and custom engines may be useful later, but they are too heavy for the first physics milestone. Phase 7 should optimize for learning value, reproducibility, and a small project footprint.

## 4. Phase 7 Subphases

### Phase 7.0: Physics Simulation Planning

Status: this document.

- Plan architecture.
- Choose simulator.
- Define scope.
- Define out-of-scope items.
- Define testing strategy.
- No implementation.
- Do not add PyBullet yet.
- Do not add dependencies yet.
- Do not modify source logic yet.

### Phase 7.1: PyBullet Setup Spike

Status: started in `docs/phase_7_pybullet_setup.md`.

- Add PyBullet as an optional dependency.
- Create a minimal empty physics world.
- Verify PyBullet installation and execution on Windows.
- Keep the spike small and reversible.
- Document setup behavior and limitations.

### Phase 7.2: Single Robot Body

Status: planned.

- Add a simple robot body, such as a box, cylinder, or capsule-like shape.
- Track basic position and orientation.
- Add simple movement commands.
- Keep the robot model intentionally simple.
- Avoid complex control theory or motor modeling.

### Phase 7.3: Obstacles And Goals

Status: planned.

- Add static obstacles.
- Add visible goal markers.
- Detect when a robot reaches a goal region.
- Keep goal detection simple and testable.
- Avoid complex navigation stacks.

### Phase 7.4: Multi-Robot Physics Scene

Status: planned.

- Add multiple robot bodies to the same physics world.
- Detect robot-robot and robot-obstacle collisions.
- Support simple simultaneous movement.
- Preserve the idea of multi-agent coordination without copying grid-world internals.

### Phase 7.5: Physics-Based Demo And Logging

Status: planned.

- Run a simple physics demo.
- Log robot positions.
- Log collisions.
- Log goal success or failure.
- Keep output readable and portfolio-friendly.

### Phase 7.6: Final Phase 7 Polish

Status: planned.

- Update documentation.
- Add tests where practical.
- Keep existing tests passing.
- Write `docs/phase_7_summary.md`.
- Prepare a Phase 7 milestone tag.

## 5. Proposed Architecture

Physics simulation should live in a separate package namespace from the existing grid-world system.

Proposed future files:

```text
src/marlsim/physics/__init__.py
src/marlsim/physics/world.py
src/marlsim/physics/robot.py
src/marlsim/physics/scenarios.py
src/marlsim/demos/physics_demo.py
tests/test_physics_world.py
docs/phase_7_summary.md
```

Expected responsibilities:

- `world.py`: PyBullet connection setup, world reset, stepping, gravity, plane creation, and collision queries.
- `robot.py`: simple robot body creation, pose access, and basic movement commands.
- `scenarios.py`: small reusable physics scenes with obstacles, robot starts, and goals.
- `physics_demo.py`: runnable beginner-friendly demo for the physics milestone.
- `test_physics_world.py`: lightweight tests for pure helper logic and import-safe behavior where practical.
- `phase_7_summary.md`: final summary once Phase 7 is complete.

## 6. Design Principle

Physics simulation should be separate from the existing grid-world system.

Rules:

- Do not rewrite `GridWorldEnv`.
- Do not break previous phases.
- Do not change grid-world movement, collision, reward, or communication behavior to fit physics.
- Do not make PyBullet a requirement for running existing grid-world demos or tests.
- Keep physics code optional and isolated behind `marlsim.physics`.
- Reuse ideas from earlier phases, but do not force the physics simulator to mimic grid-world internals.

This preserves the project as a clean learning sequence: symbolic coordination first, then a separate physical simulation path.

## 7. Testing Strategy

Phase 7 should keep testing pragmatic.

Testing goals:

- Keep all existing tests passing.
- Add lightweight tests for pure helper logic.
- Test configuration, scenario definitions, simple data conversion, and goal-distance helpers where possible.
- Avoid brittle graphical tests.
- Avoid tests that depend heavily on exact physics engine timing, rendering, or floating-point contact details.
- Prefer deterministic direct-mode PyBullet tests only if they are stable on Windows and local development machines.

The project should not gain a fragile test suite just because a physics engine is involved. Physics demos can provide manual validation, while unit tests should focus on stable logic.

## 8. Out Of Scope

Phase 7 should not include:

- ROS.
- Gazebo.
- Real robot deployment.
- Deep reinforcement learning.
- Continuous-control learning.
- Advanced sensors.
- Camera perception.
- Sim-to-real transfer.
- Complex robot URDF modeling.
- SLAM.
- Path planning stacks.
- Distributed robotics systems.

These topics are valuable future directions, but they would make the first physics milestone too broad and too difficult to maintain.

## 9. GitHub And Portfolio Note

Phase 7 should eventually produce visual material for the portfolio.

The final physics demo should be suitable for screenshots, GIFs, or short video recordings. That material can support Phase 8 by showing the project progression from grid-world coordination to a simple physical robotics scene.

The first priority is still correctness and clarity. Recording material should come after the basic physics world, robot body, obstacles, goals, logging, and documentation are stable.

## 10. Risks

### Dependency Complexity

PyBullet is lightweight compared with ROS or Gazebo, but it is still an external simulator dependency.

Mitigation:

- Add it only in Phase 7.1.
- Make it optional.
- Keep existing grid-world tests independent from PyBullet.

### Architecture Drift

Physics code could become tangled with grid-world code if the project tries to reuse too much too quickly.

Mitigation:

- Add `marlsim.physics` as a separate namespace.
- Keep `GridWorldEnv` unchanged.
- Share concepts through documentation and demos before sharing code.

### Brittle Tests

Physics engines can produce small platform-specific differences.

Mitigation:

- Test stable helper logic first.
- Use broad tolerances where physics tests are necessary.
- Avoid graphical assertions.

### Scope Creep

Physics simulation can easily lead to ROS, real robots, advanced control, sensors, and deep RL.

Mitigation:

- Keep Phase 7 focused on a simple PyBullet scene.
- Defer advanced robotics and learning topics to later phases.

## 11. Success Criteria

Phase 7 should be considered successful when:

- PyBullet is added as an optional dependency.
- A minimal physics world can run on Windows.
- A simple robot body can move with basic commands.
- Static obstacles and goal markers exist.
- A small multi-robot scene can run with collision detection.
- A demo logs positions, collisions, and success.
- Existing grid-world tests and demos remain available.
- Documentation explains what changed and what remains out of scope.

Phase 7.0 is successful when the plan is documented and the project clearly identifies Phase 7.1 as the next implementation milestone.

## 12. Next Implementation Step

The next milestone is Phase 7.1: PyBullet setup spike.

Phase 7.1 should add the optional PyBullet dependency, create a minimal empty physics world, and verify that the setup works on Windows without changing the existing grid-world simulation behavior.
