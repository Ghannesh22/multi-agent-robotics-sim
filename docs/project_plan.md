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

Status: next. BFS and `ShortestPathAgent` exist now as Phase 1 baseline tools and as a seed for Phase 2, not as full Phase 2 completion.

- Greedy goal-seeking agent.
- Breadth-first shortest-path agent.
- Same-cell conflict handling.
- Priority conflict policy available for experiments.
- Future coordination scenarios with bottlenecks, crossings, and narrow corridors.

### Phase 3: Gymnasium-Style Reinforcement Learning Environment

Not started.

- Observation space.
- Action space.
- Reward function.
- Episode reset.
- Termination conditions.

### Phase 4: Single-Agent Reinforcement Learning Baseline

Not started.

- Train one agent first.
- Compare learned behavior with random and rule-based behavior.

### Phase 5: Multi-Agent Reinforcement Learning

Not started.

- Multiple learning agents.
- Shared environment.
- Cooperation and competition.
- Collision penalties.
- Goal rewards.

### Phase 6: Communication Between Agents

Not started.

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

This milestone is implemented as the initial project slice. BFS and `ShortestPathAgent` are already present as baseline tools and as a seed for later Phase 2 coordination work. Phase 1 is complete. The next milestone is Phase 2: rule-based coordination.

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
