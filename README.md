# Multi-Agent Robotics Simulation with Learning-Based Coordination

A simulation-first personal machine learning project for learning multi-agent robotics, coordination, and reinforcement learning from the ground up.

This first milestone intentionally avoids ROS, Gazebo, PyBullet, Unity, and deep reinforcement learning. It starts with a clean 2D grid-world simulator that can later grow into rule-based coordination, Gymnasium-style RL environments, and multi-agent learning experiments.

## Current Milestone

Phase 6 is complete and documented in `docs/phase_6_summary.md`. Phase 7 planning has started in `docs/phase_7_plan.md`, and the Phase 7.1 PyBullet setup spike is documented in `docs/phase_7_pybullet_setup.md`. The project now has a stable grid-world foundation, rule-based coordination policies, single-agent Q-learning, lightweight multi-agent tabular learning, explicit symbolic communication experiments with evaluation and trajectory visualization, and an isolated optional physics setup path.

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

Completed in Phase 5:

- Phase 5.0 planning roadmap in `docs/phase_5_plan.md`.
- `MultiAgentRLEnv` wrapper around the existing `GridWorldEnv`.
- Independent multi-agent Q-learning with one Q-table per controlled agent.
- Individual and shared/team reward modes.
- MARL evaluation against learned policies and rule-based baselines.
- MARL metrics for success, timeout, blocked moves, rewards, and episode length.
- Text trajectory visualization for learned behavior.
- Demos showing individual reward success and shared reward failure.
- Phase 5 completion summary in `docs/phase_5_summary.md`.

Completed in Phase 6:

- Phase 6.0 planning roadmap in `docs/phase_6_plan.md`.
- Communication motivation from the Phase 5 shared reward coordination failure.
- Initial communication message vocabulary: intended next move, current target, blocked status, wait signal, and priority signal.
- Phase 6 subphase roadmap from message design through communication evaluation and visualization.
- Explicit scope boundary excluding natural language, LLM agents, deep MARL, centralized critics, external MARL frameworks, ROS, and physics simulation.
- Phase 6.1 message design in `docs/phase_6_message_design.md`.
- Lightweight `CommunicationMessage` dataclass with formatting, serialization, and validation helpers.
- Phase 6.2 communication-enabled MARL wrapper in `docs/phase_6_communication_wrapper.md`.
- `CommunicatingMultiAgentRLEnv` with optional `step(actions, messages)` support and communication info logging.
- Phase 6.3 rule-based communication experiments in `docs/phase_6_rule_based_communication.md`.
- `CommunicationAwareWaitingAgent` and `CommunicationAwarePriorityAgent` for deterministic message-aware coordination.
- A communication demo comparing no-communication and communication-aware priority behavior.
- Phase 6.4 communication-aware Q-learning in `docs/phase_6_communication_q_learning.md`.
- `CommunicationAwareQLearningAgent`, communication-enhanced observations, and `train_communication_q_learning`.
- A comparison demo for independent MARL vs message-augmented tabular Q-learning.
- Phase 6.5 communication evaluation in `docs/phase_6_communication_evaluation.md`.
- Common metrics for independent MARL, rule-based communication, and communication-aware Q-learning.
- A communication evaluation demo comparing success, timeouts, blocked moves, rewards, episode length, message counts, predicted conflicts, and waiting frequency.
- Phase 6.6 communication visualization in `docs/phase_6_communication_visualization.md`.
- Text trajectory visualization for independent MARL, rule-based communication, and communication-aware Q-learning behavior.
- Message timelines, waiting traces, conflict prediction traces, and rollout summaries.
- Phase 6 completion summary in `docs/phase_6_summary.md`.

Started in Phase 7:

- Phase 7.0 planning roadmap in `docs/phase_7_plan.md`.
- Phase 7.1 PyBullet setup spike in `docs/phase_7_pybullet_setup.md`.
- Recommended simulator: PyBullet, introduced as an optional dependency.
- Architecture plan for keeping physics simulation separate from the existing grid-world system.
- Explicit boundary that Phase 7 should not rewrite `GridWorldEnv` or break previous phases.
- Minimal `PhysicsWorld` helper for DIRECT-mode connection, plane creation, stepping, and clean disconnect.

Phase 6 stayed symbolic, tabular, and dependency-free. It did not add deep RL, neural networks, PyTorch, TensorFlow, RLlib, PettingZoo, ROS, physics simulation, natural language communication, LLM agents, or changes to `GridWorldEnv` movement rules.

Phase 7 physics support is optional and isolated. It has not added robots, obstacles, goals, RL integration, continuous-control learning, ROS, Gazebo, or changes to the existing grid-world behavior.

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

## Run MARL Demos

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_training_demo
python -m marlsim.demos.marl_reward_comparison
python -m marlsim.demos.marl_evaluation_comparison
python -m marlsim.demos.marl_visualization_demo
```

These demos train independent multi-agent Q-learning policies, compare individual and shared rewards, evaluate learned policies against baselines, and print text trajectories for learned behavior.

## Run Communication Demo

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.communication_demo
python -m marlsim.demos.communication_q_learning_demo
python -m marlsim.demos.communication_evaluation_demo
python -m marlsim.demos.communication_visualization_demo
```

These demos compare a no-communication shortest-path step against a communication-aware priority step, compare independent MARL against message-augmented tabular Q-learning, print a shared evaluation table, and show readable communication trajectories with messages, waiting, priorities, predicted conflicts, actions, rewards, and blocked moves.

## Run Physics Setup Demo

Install the optional Phase 7 physics dependency first:

```powershell
python -m pip install -e ".[physics]"
```

Then run the DIRECT-mode PyBullet setup demo:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.physics_setup_demo
```

This creates a simple PyBullet plane, runs a few simulation steps, prints a success message, and disconnects cleanly. It does not add robot bodies, obstacles, goals, or RL integration yet.

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

## Phase 5 Plan

See [docs/phase_5_plan.md](docs/phase_5_plan.md) for the multi-agent reinforcement learning plan. Phase 5 includes the multi-agent wrapper, independent Q-learning, cooperative reward modes, MARL evaluation helpers, and text trajectory demos.

See [docs/phase_5_marl_evaluation.md](docs/phase_5_marl_evaluation.md) for the MARL evaluation design and comparison metrics.

See [docs/phase_5_marl_demos.md](docs/phase_5_marl_demos.md) for the MARL behavior demo and trajectory output format.

See [docs/phase_5_summary.md](docs/phase_5_summary.md) for a concise review of the completed multi-agent reinforcement learning milestone, experiment results, out-of-scope items, and why Phase 6 communication comes next.

## Phase 6 Plan

See [docs/phase_6_plan.md](docs/phase_6_plan.md) for the communication planning milestone. Phase 6.0 defines why communication matters, why shared rewards were insufficient, the initial message vocabulary, the planned communication subphases, and the out-of-scope boundaries.

See [docs/phase_6_message_design.md](docs/phase_6_message_design.md) for the Phase 6.1 structured communication message design.

See [docs/phase_6_communication_wrapper.md](docs/phase_6_communication_wrapper.md) for the Phase 6.2 wrapper design.

See [docs/phase_6_rule_based_communication.md](docs/phase_6_rule_based_communication.md) for the Phase 6.3 deterministic communication policy design.

See [docs/phase_6_communication_q_learning.md](docs/phase_6_communication_q_learning.md) for the Phase 6.4 message-augmented tabular Q-learning design.

See [docs/phase_6_communication_evaluation.md](docs/phase_6_communication_evaluation.md) for the Phase 6.5 communication comparison methodology and findings.

See [docs/phase_6_communication_visualization.md](docs/phase_6_communication_visualization.md) for the Phase 6.6 text trajectory visualization design.

See [docs/phase_6_summary.md](docs/phase_6_summary.md) for a concise review of the completed communication milestone, experiment results, out-of-scope items, and why Phase 7 physics-based simulation comes next.

## Phase 7 Plan

See [docs/phase_7_plan.md](docs/phase_7_plan.md) for the physics-based simulation planning milestone. Phase 7.0 explains why moving beyond grid-world abstraction matters, recommends PyBullet as the first simulator, defines the Phase 7 subphases, proposes a separate `marlsim.physics` architecture, and keeps ROS, Gazebo, real robot deployment, deep RL, continuous-control learning, advanced sensors, camera perception, and sim-to-real transfer out of scope.

See [docs/phase_7_pybullet_setup.md](docs/phase_7_pybullet_setup.md) for the Phase 7.1 PyBullet setup spike. It documents the optional physics dependency, why DIRECT mode is used first, how to run the setup demo, and what remains intentionally out of scope.

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
9. Physics-based simulator upgrade.
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

Phase 1 was completed and tagged as `v0.2.0-phase-1-complete`. Phase 2 was completed and tagged as `v0.3.0-phase-2-complete`. Phase 3 was completed and tagged as `v0.4.0-phase-3-rl-env-complete`. The recommended Phase 4 completion tag is `v0.5.0-phase-4-q-learning-complete`. Phase 6 is complete; Phase 7.0 planning has started and Phase 7.1 adds the first optional PyBullet setup spike.
