# Multi-Agent Robotics Simulation with Learning-Based Coordination

A simulation-first personal machine learning project for learning multi-agent robotics, coordination, and reinforcement learning from the ground up.

This first milestone intentionally avoids ROS, Gazebo, PyBullet, Unity, and deep reinforcement learning. It starts with a clean 2D grid-world simulator that can later grow into rule-based coordination, Gymnasium-style RL environments, and multi-agent learning experiments.

## Current Milestone

Phase 1.0 and Phase 1.1 are complete.

Implemented in Phase 1.0:

- 2D grid-world environment.
- Multiple agents.
- Obstacles and per-agent goals.
- Step-by-step simultaneous movement.
- Collision prevention for walls, obstacles, same-cell conflicts, and direct swaps.
- Basic console renderer.
- Random, greedy, and shortest-path rule-based agents.
- BFS path planning as a Phase 1 baseline tool and Phase 2 seed.
- A small runnable demo.
- Unit tests for core movement and collision behavior.

Completed in Phase 1.1:

- Architecture and code review.
- Cleanup needs documented in `docs/code_review.md`.
- Next milestone clarified.

Next step:

- Phase 1.2: improve the text-based visualization while keeping the simulator simple.

## Run the Demo

From this directory:

```powershell
python -m marlsim.demos.basic_grid
```

If running directly from a fresh checkout without installing the package:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.basic_grid
```

## Run Tests

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
```

## Project Direction

The planned progression is:

1. Basic grid-world simulation and documentation review.
2. Rule-based coordination and path planning.
3. Gymnasium-style reinforcement learning interface.
4. Single-agent RL baseline.
5. Multi-agent RL.
6. Communication between agents.
7. Optional physics-based simulator upgrade.
8. Portfolio demo and final report.

See [docs/project_plan.md](docs/project_plan.md) for the full roadmap.

## Version Control Workflow

This project should use Git from the beginning.

- Keep `main` stable and milestone-ready.
- Use focused branches for larger features, such as `feature/environment`, `feature/agents`, `feature/planning`, `feature/rl-env`, and `feature/portfolio-demo`.
- Write meaningful commit messages that describe the behavior or documentation change.
- Push progress to GitHub after each milestone.
- Track future work with GitHub Issues or a small TODO board.
- Keep the README current as the project evolves.

Phase 1.0 and Phase 1.1 should be pushed with the grid-world simulator, collision rules, starter agents, BFS planner, console demo, tests, README, and docs. BFS and `ShortestPathAgent` are included as baseline tools for Phase 1 and as seeds for Phase 2 coordination work, not as full Phase 2 completion. The recommended milestone tag is `v0.1.0-phase-1-grid-world`.
