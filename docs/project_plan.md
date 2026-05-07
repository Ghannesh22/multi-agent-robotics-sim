# Multi-Agent Robotics Simulation with Learning-Based Coordination

## Vision

Build a simulation-first multi-agent robotics system where multiple agents move through an environment, avoid collisions, reach goals, coordinate with each other, and later learn better behavior using machine learning.

## Long-Term Goal

The final version should include a clean grid-world simulator, rule-based coordination, a Gymnasium-style reinforcement learning interface, single-agent and multi-agent RL experiments, optional agent communication, and a portfolio-ready demo with documentation and result comparisons.

## Roadmap

### Phase 1.0: Basic Grid-World Simulation

Status: completed.

- Multiple agents.
- Obstacles.
- Goals.
- Step-by-step movement.
- Collision prevention.
- Basic console visualization.
- Random, greedy, and shortest-path baseline agents.
- BFS path planning as a baseline tool.

### Phase 1.1: Architecture And Code Review

Status: completed.

- Current architecture reviewed.
- Cleanup needs documented.
- Improvement suggestions recorded in `docs/code_review.md`.
- Scope kept documentation-only with no new simulator features.

### Phase 1.2: Better Text-Based Visualization

Status: completed.

- Improve the readability of the existing text output.
- Keep visualization lightweight and beginner-friendly.
- Do not add complex graphics before simulation rules are stable.

### Phase 1.3: Scenario Configuration

Status: completed.

- Make scenarios easier to define, load, and compare.
- Keep scenario configuration separate from environment logic.
- Preserve the current grid-world mechanics while making demo inputs more flexible.
- Add reusable scenario builders and named demo scenarios.

### Phase 1.4: Experiment Logging

Status: completed.

- Add lightweight experiment logging for demo and scenario runs.
- Record scenario name, step count, agent outcomes, and basic run metadata.
- Keep logging simple and text-based; do not introduce databases or dashboards.
- Print final run summaries from the demo.

### Phase 1.5: Final Phase 1 Polish And Stabilization

Status: completed.

- Confirm Phase 1 is stable before starting coordination work.
- Review scenarios, demos, tests, and documentation together.
- Reserve final cleanup, naming consistency, and polish for this milestone.
- Add a concise Phase 1 completion summary.

### Phase 2: Rule-Based Coordination

Status: completed. See `docs/phase_2_plan.md`, `docs/phase_2_conflict_aware_waiting.md`, `docs/phase_2_priority_coordination.md`, `docs/phase_2_local_replanning.md`, `docs/phase_2_metrics.md`, and `docs/phase_2_summary.md`.

- Phase 2.1: conflict-aware waiting. Completed with `ConflictAwareWaitingAgent`.
- Phase 2.2: priority-based coordination. Completed with `PriorityBasedCoordinationAgent`.
- Phase 2.3: local replanning. Completed with `LocalReplanningAgent`.
- Phase 2.4: coordination metrics. Completed with a repeatable comparison runner.
- Phase 2.5: final Phase 2 polish, documentation, tests, scenario comparisons, and tag preparation. Completed.

### Phase 3: Gymnasium-Style Reinforcement Learning Environment

Status: completed. See `docs/phase_3_plan.md`, `docs/phase_3_observation_design.md`, `docs/phase_3_action_design.md`, `docs/phase_3_reward_design.md`, `docs/phase_3_episode_handling.md`, `docs/phase_3_manual_validation.md`, and `docs/phase_3_summary.md`.

- Phase 3.0: RL environment planning. Completed.
- Phase 3.1: RL wrapper skeleton around `GridWorldEnv`. Completed with `SingleAgentRLEnv`.
- Phase 3.2: observation design. Completed with `(agent_x, agent_y, goal_x, goal_y)`.
- Phase 3.3: action design. Completed with five integer actions mapped to `Action`.
- Phase 3.4: reward design. Completed with simple documented reward values.
- Phase 3.5: episode handling. Completed with explicit done, timeout, and info behavior.
- Phase 3.6: manual validation. Completed with `marlsim.demos.rl_env_validation`.
- Phase 3.7: final Phase 3 polish, documentation, tests, and validation. Completed.

### Phase 4: Single-Agent Reinforcement Learning Baseline

Status: completed. See `docs/phase_4_plan.md`, `docs/phase_4_q_learning.md`, `docs/phase_4_training_loop.md`, `docs/phase_4_evaluation.md`, `docs/phase_4_metrics.md`, and `docs/phase_4_summary.md`.

- Phase 4.0: RL training planning. Completed.
- Phase 4.1: Q-table agent. Completed with `QLearningAgent`.
- Phase 4.2: training loop. Completed with `train_q_learning`.
- Phase 4.3: evaluation and comparison. Completed with random, greedy, shortest-path, and Q-learning comparison.
- Phase 4.4: training metrics and visualization. Completed with text summaries, ASCII charts, and CSV export.
- Phase 4.5: final Phase 4 polish, documentation, tests, and demos. Completed.

### Phase 5: Multi-Agent Reinforcement Learning

Status: completed. See `docs/phase_5_plan.md`, `docs/phase_5_multi_agent_wrapper.md`, `docs/phase_5_independent_q_learning.md`, `docs/phase_5_cooperative_rewards.md`, `docs/phase_5_marl_evaluation.md`, `docs/phase_5_marl_demos.md`, and `docs/phase_5_summary.md`.

- Phase 5.0: MARL planning. Completed.
- Phase 5.1: multi-agent RL wrapper. Completed with `MultiAgentRLEnv`.
- Phase 5.2: independent Q-learning agents. Completed with decentralized Q-table training.
- Phase 5.3: cooperative reward experiments. Completed with individual and shared reward modes.
- Phase 5.4: MARL evaluation and metrics. Completed with learned-policy and baseline comparison helpers.
- Phase 5.5: MARL visualization and demos. Completed with text trajectory output.
- Phase 5.6: final Phase 5 polish, documentation, tests, demos, and tag preparation. Completed.

### Phase 6: Communication Between Agents

Next implementation milestone.

- Share position.
- Share intent.
- Negotiate movement.
- Compare communication vs no communication.

### Phase 7: Physics-Based Simulation Upgrade

Not started.

- Move from grid-world to PyBullet or similar.
- Use simple robot bodies.
- Add continuous movement.
- Add sensors.

### Phase 8: Portfolio Demo

Not started.

- Clean GitHub repository.
- Demo video.
- Documentation.
- Final report.
- Results comparison.

## First Milestone

Build a 2D grid-world simulator where two agents can move step-by-step toward goals while avoiding obstacles and collisions.

This milestone is implemented as the initial project slice. BFS and `ShortestPathAgent` are already present as baseline tools and as a seed for later coordination work. Phase 1, Phase 2, Phase 3, and Phase 4 are complete. The next implementation milestone is Phase 5: multi-agent reinforcement learning.

## Recommended Executive Roles

- Planning executive: maintains scope, roadmap, and milestones.
- Environment executive: owns grid state, stepping, reset, and scenario setup.
- Simulation logic executive: owns collision prevention, conflict resolution, and movement rules.
- Agent logic executive: owns random, greedy, path-planning, and later communication agents.
- ML/RL executive: owns Gymnasium-style APIs, rewards, training, and evaluation.
- Testing executive: owns unit tests and edge-case coverage.
- Documentation executive: owns README, architecture notes, demos, and reports.

## What Not To Do Yet

- Do not start with ROS, Gazebo, PyBullet, Unity, or complex robotics middleware.
- Do not begin with deep reinforcement learning.
- Do not train multiple learning agents before a single-agent baseline works.
- Do not build fancy graphics before simulation rules are correct.
- Do not mix simulation logic, rendering, and ML training in one large file.

## 10. Version Control and GitHub Workflow

Use Git from the beginning so the project history tells a clear portfolio story.

### Repository Setup

- Initialize a Git repository in `multi-agent-robotics-sim`.
- Keep this project separate from unrelated repositories.
- Commit the Phase 1 scaffold as the first meaningful project commit.
- Add a GitHub remote once the repository exists.
- Push progress after each milestone.

Recommended first commit:

```text
Initialize grid-world simulation project
```

Recommended Phase 1 tag:

```text
v0.1.0-phase-1-grid-world
```

### Branch Strategy

Keep `main` stable and use short-lived feature branches for major work:

- `feature/environment`
- `feature/agents`
- `feature/planning`
- `feature/rl-env`
- `feature/single-agent-rl`
- `feature/multi-agent-rl`
- `feature/communication`
- `feature/portfolio-demo`

Merge branches only after tests pass and the README or docs are updated when behavior changes.

### Commit Message Style

Use concise messages that describe the actual change:

- `Add simultaneous multi-agent movement`
- `Add collision prevention tests`
- `Add BFS path planning baseline`
- `Document Phase 1 milestone`
- `Add Gymnasium-style environment skeleton`

Prefer several focused commits over one large unclear commit.

### Issue and TODO Tracking

Use GitHub Issues for upcoming work:

- Phase 2 rule-based coordination.
- Path planning improvements.
- Gymnasium-style RL environment.
- Single-agent RL baseline.
- Multi-agent RL prototype.
- Communication experiments.
- Portfolio demo assets.

Suggested labels:

- `phase-1`
- `phase-2`
- `rl`
- `testing`
- `docs`
- `demo`

### What To Push At The End Of Phase 1

Push all working Phase 1 materials:

- Core grid-world environment.
- Multiple-agent movement.
- Obstacles, goals, and step-by-step simulation.
- Collision prevention for walls, obstacles, same-cell conflicts, direct swaps, and stationary agents.
- Random, greedy, and shortest-path agents.
- BFS path planner as a Phase 1 baseline and Phase 2 seed.
- Console renderer.
- Named scenario configuration for simple, narrow corridor, and crossing paths demos.
- Simple experiment summaries for demo runs.
- Phase 1 completion summary in `docs/phase_1_summary.md`.
- Basic runnable demo.
- Unit tests for movement, collision rules, agents, and planning.
- `README.md`, `docs/project_plan.md`, `docs/architecture.md`, `docs/code_review.md`, `.gitignore`, and `pyproject.toml`.

Before pushing, verify:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
python -m marlsim.demos.basic_grid
```

### What To Document In GitHub

Keep the GitHub repository understandable to someone seeing it for the first time.

README should include:

- What the project is.
- Why it starts with grid-world simulation.
- Current milestone status.
- How to run the demo.
- How to run tests.
- Roadmap summary.
- Demo GIF or screenshot later.

Docs should include:

- `docs/project_plan.md`: roadmap, milestones, and GitHub workflow.
- `docs/architecture.md`: core modules and design decisions.
- Future `docs/experiments.md`: RL experiments, metrics, plots, and comparisons.

Demos should include:

- A basic console demo now.
- Later, a GIF or video of agents moving.
- Later, experiment plots comparing random, rule-based, and learned behavior.

### Portfolio Quality Guidelines

Make the repository look strong by keeping it clear, incremental, and reproducible.

- Keep project structure readable.
- Maintain a clean README.
- Show steady progress through meaningful commits.
- Add tests with every core behavior change.
- Include result comparisons as the ML phases begin.
- Add demo media once visualization improves.
- Keep beginner-friendly explanations without hiding technical depth.
