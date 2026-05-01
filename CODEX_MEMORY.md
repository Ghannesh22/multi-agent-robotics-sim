# Multi-Agent Robotics Simulation Memory

This is the project-specific memory for `D:\D\projects\multi-agent-robotics-sim`.

## Project Identity

- Title: Multi-Agent Robotics Simulation with Learning-Based Coordination.
- Goal: Build a simulation-first multi-agent robotics project that starts with a clean 2D grid-world and gradually grows toward coordination, reinforcement learning, communication, and optional physics simulation.
- Current constraint: keep the project simple. Do not add ROS, Gazebo, PyBullet, Unity, or deep reinforcement learning until the grid-world foundation is stable.

## Current State

- Phase 1 grid-world implementation is complete and frozen as `v0.1.0`.
- Git repository exists on branch `main`.
- Phase 1 commit: `9359c13 Initialize grid-world simulation project`.
- Tags:
  - `v0.1.0`
  - `v0.1.0-phase-1-grid-world`
- No GitHub remote is configured yet.
- Post-freeze documentation commit adds `docs/code_review.md` and this memory file.

## Implemented Phase 1 Capabilities

- Standard-library-only Python package under `src/marlsim`.
- Grid-world environment with fixed width and height.
- Multiple agents with per-agent goals.
- Obstacles.
- Discrete actions: up, down, left, right, stay.
- Simultaneous environment stepping.
- Collision prevention for out-of-bounds moves, obstacle moves, same-cell conflicts, direct swaps, and moves into stationary agents.
- Console renderer.
- Random, greedy, and BFS shortest-path agents.
- Basic demo: `python -m marlsim.demos.basic_grid`.
- Unit tests for environment, agents, and planning.

## Verification

Use these commands from the project root:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
python -m marlsim.demos.basic_grid
```

Last verified before the initial commit:

- 13 tests passed.
- Demo reached both goals in 6 steps.

## Important Decisions

- Keep simulation logic separate from agent logic, planning, visualization, scenarios, demos, tests, and docs.
- Use deterministic, conservative collision rules before adding learning.
- Use rule-based baselines before reinforcement learning.
- Use `unittest` for now to avoid unnecessary dependencies.
- Maintain Git history and documentation from the beginning for portfolio quality.

## Next Safe Milestone

Next milestone should be Phase 2.1: Coordination Scenarios And Rule-Based Baselines.

Recommended scope:

- Add deterministic coordination scenarios such as narrow corridors, crossing paths, and shared bottlenecks.
- Add tests for conflict resolution and blocked-path edge cases.
- Add simple metrics for goals completed, blocked moves, prevented collisions, and total steps.
- Keep agents rule-based and transparent.

Do not add reinforcement learning, PyBullet, or new simulator complexity until the user explicitly asks.
