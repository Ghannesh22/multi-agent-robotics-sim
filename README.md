# Multi-Agent Robotics Simulation with Learning-Based Coordination

A simulation-first personal machine learning project for learning multi-agent robotics, coordination, and reinforcement learning from the ground up.

This first milestone intentionally avoids ROS, Gazebo, PyBullet, Unity, and deep reinforcement learning. It starts with a clean 2D grid-world simulator that can later grow into rule-based coordination, Gymnasium-style RL environments, and multi-agent learning experiments.

## Current Milestone

Phase 4 is complete. The project now has a stable grid-world foundation, rule-based coordination policies, a single-agent RL-style environment wrapper, and a first tabular Q-learning training/evaluation milestone.

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

Completed in Phase 1.2:

- Cleaner text-based grid rendering.
- Clear cell borders.
- Distinct symbols for agents, goals, obstacles, and empty cells.
- Step number and agent position display.
- Legend included with each rendered grid.

Completed in Phase 1.3:

- Simple Python scenario configuration helpers.
- Reusable scenario builders for grid dimensions, agents, goals, and obstacles.
- Named scenarios: `simple`, `narrow_corridor`, and `crossing_paths`.
- Demo scenario selection from the command line.

Completed in Phase 1.4:

- Simple non-learning experiment logging.
- Final run summaries for scenario name, result, total steps, reached goals, blocked moves, and final positions.
- Blocked move counts from existing step event data.

Completed in Phase 1.5:

- Final README and roadmap polish.
- Phase 1 completion summary in `docs/phase_1_summary.md`.
- Full test and demo verification across all starter scenarios.

Completed in Phase 2:

- Phase 2.0 planning roadmap in `docs/phase_2_plan.md`.
- Conflict-aware waiting with `ConflictAwareWaitingAgent`.
- Priority-based coordination with `PriorityBasedCoordinationAgent`.
- Local replanning with `LocalReplanningAgent`.
- Repeatable coordination metrics across all current scenarios.
- Phase 2 completion summary in `docs/phase_2_summary.md`.

Completed in Phase 3:

- Phase 3.0 planning roadmap in `docs/phase_3_plan.md`.
- `SingleAgentRLEnv` wrapper around the existing `GridWorldEnv`.
- Simple coordinate-vector observation: `(agent_x, agent_y, goal_x, goal_y)`.
- Five integer actions mapped to the existing `Action` enum.
- Simple documented reward function.
- Explicit episode lifecycle, done, timeout, and info behavior.
- Manual RL wrapper validation demo.
- Phase 3 completion summary in `docs/phase_3_summary.md`.

Completed in Phase 4:

- Phase 4.0 planning roadmap in `docs/phase_4_plan.md`.
- `QLearningAgent` with tabular Q-values.
- Epsilon-greedy exploration.
- Single-agent Q-learning training loop.
- Policy evaluation against random, greedy, shortest-path, and trained Q-learning policies.
- Training metrics, text summaries, ASCII-style charts, and CSV export.
- Lightweight Q-learning training, comparison, and metrics demos.
- Phase 4 completion summary in `docs/phase_4_summary.md`.

Next step after Phase 4:

- Phase 5: multi-agent reinforcement learning.

Phase 4 intentionally did not add deep RL, Gymnasium, Stable-Baselines3, PyTorch, TensorFlow, PyBullet, ROS, physics simulation, multi-agent RL, communication learning, or neural networks.

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

To choose a scenario:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.basic_grid --scenario narrow_corridor
python -m marlsim.demos.basic_grid --scenario crossing_paths
```

The demo prints a plain-text grid after each step. The renderer uses these symbols:

- `A#`: agent, such as `A1` or `A2`.
- `A#*`: agent currently on its goal.
- `G#`: goal for the matching agent.
- `G+`: shared goal cell.
- `###`: obstacle.
- `.`: empty cell.

Each render also shows the step number, a legend, and each agent's current position and goal.
At the end of the run, the demo prints a final run summary with total steps, success or failure, reached goals, blocked moves, and final agent positions.

## Run Tests

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
```

## Run Coordination Comparison

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.coordination_comparison
```

This prints a table comparing the baseline shortest-path agent, conflict-aware waiting, priority coordination, and local replanning across `simple`, `narrow_corridor`, and `crossing_paths`.

## Run RL Wrapper Validation

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.rl_env_validation
```

This prints a scripted validation of the Phase 3 single-agent RL wrapper: reset, normal movement, blocked movement, goal completion, and timeout behavior. It is not RL training.

## Run Q-Learning Demos

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_demo
python -m marlsim.demos.q_learning_comparison
python -m marlsim.demos.q_learning_metrics_demo
```

These demos train a small tabular Q-learning agent, compare it with rule-based baselines, and print dependency-free training metrics and text charts.

## Phase 1 Summary

See [docs/phase_1_summary.md](docs/phase_1_summary.md) for a concise review of what Phase 1 achieved, available scenarios, logged metrics, out-of-scope items, and why Phase 2 comes next.

## Phase 2 Summary

See [docs/phase_2_summary.md](docs/phase_2_summary.md) for a concise review of what Phase 2 achieved, the implemented coordination policies, comparison results, out-of-scope items, and why Phase 3 comes next. See [docs/phase_2_plan.md](docs/phase_2_plan.md) for the original rule-based coordination roadmap.

## Phase 3 Summary

See [docs/phase_3_summary.md](docs/phase_3_summary.md) for a concise review of what Phase 3 achieved, the RL wrapper contract, validation results, out-of-scope items, and why Phase 4 training comes next. See [docs/phase_3_plan.md](docs/phase_3_plan.md) for the original Phase 3 roadmap.

## Phase 4 Plan

See [docs/phase_4_plan.md](docs/phase_4_plan.md) for the single-agent reinforcement learning plan. Phase 4.0 is planning-only and recommends tabular Q-learning before deep reinforcement learning.

## Phase 4 Summary

See [docs/phase_4_summary.md](docs/phase_4_summary.md) for a concise review of the first tabular Q-learning milestone, training/evaluation metrics, demos, out-of-scope items, and why Phase 5 comes next.

## Project Direction

The planned progression is:

1. Basic grid-world simulation, documentation review, visualization, and scenario configuration.
2. Experiment logging in Phase 1.4.
3. Final Phase 1 polish and stabilization in Phase 1.5.
4. Rule-based coordination and path planning.
5. Gymnasium-style reinforcement learning interface, starting with a single controlled agent wrapper.
6. Single-agent RL baseline.
7. Multi-agent RL.
8. Communication between agents.
9. Optional physics-based simulator upgrade.
10. Portfolio demo and final report.

See [docs/project_plan.md](docs/project_plan.md) for the full roadmap.

## Version Control Workflow

This project should use Git from the beginning.

- Keep `main` stable and milestone-ready.
- Use focused branches for larger features, such as `feature/environment`, `feature/agents`, `feature/planning`, `feature/rl-env`, and `feature/portfolio-demo`.
- Write meaningful commit messages that describe the behavior or documentation change.
- Push progress to GitHub after each milestone.
- Track future work with GitHub Issues or a small TODO board.
- Keep the README current as the project evolves.

Phase 1 was completed and tagged as `v0.2.0-phase-1-complete`. Phase 2 was completed and tagged as `v0.3.0-phase-2-complete`. Phase 3 was completed and tagged as `v0.4.0-phase-3-rl-env-complete`. The recommended Phase 4 completion tag is `v0.5.0-phase-4-q-learning-complete`.
