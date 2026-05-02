# Phase 1.1 Code Review

Status: Phase 1.1 completed

This document reviews the current architecture and cleanup needs for Phase 1.1.
It is intentionally documentation-only: no new simulator features, reinforcement
learning, PyBullet, or ROS work is included in this phase.

## Review Summary

The project is in good shape for a beginner-friendly Phase 1 grid-world
simulator. The code is small, testable, and separated into clear packages:

- `marlsim.core` owns state, actions, environment stepping, validation, and
  collision rules.
- `marlsim.agents` owns policies that choose actions without mutating the
  environment.
- `marlsim.planning` owns reusable path-planning algorithms.
- `marlsim.scenarios` owns reusable environment configurations.
- `marlsim.visualization` owns rendering without changing simulation state.
- `marlsim.demos` owns runnable examples that wire the pieces together.
- `tests` provides executable coverage for the most important current behavior.

The strongest architectural choice is the separation between environment logic
and agent policy logic. `GridWorldEnv` does not know how actions are selected,
and agents do not directly change environment state. That keeps the simulator
easy to reason about and leaves room for later policy types without rewriting
the core environment.

## Current Architecture

### Core Simulation

Files:

- `src/marlsim/core/state.py`
- `src/marlsim/core/actions.py`
- `src/marlsim/core/environment.py`

The core package defines the simulator's basic language:

- `Position` represents grid coordinates.
- `AgentState` represents one agent's identity, position, and optional goal.
- `Action` represents the five discrete moves: up, down, left, right, stay.
- `GridWorldConfig` defines grid size, obstacles, initial agents, step limit,
  and conflict policy.
- `GridWorldEnv` owns reset, snapshot, stepping, collision prevention, and
  termination.

`GridWorldEnv.step()` is the central behavior. It:

1. Reads each agent's starting position.
2. Normalizes missing actions to `Action.STAY`.
3. Computes each requested target cell.
4. Blocks out-of-bounds and obstacle moves.
5. Blocks direct swaps.
6. Blocks moves into stationary agents.
7. Resolves same-cell conflicts.
8. Re-checks stationary-agent occupancy after conflicts can cascade.
9. Applies final positions.
10. Emits `AgentStepEvent` and `StepResult`.

This order is deterministic and conservative, which is appropriate for the
first simulator milestone.

### Agents

Files:

- `src/marlsim/agents/base.py`
- `src/marlsim/agents/random_agent.py`
- `src/marlsim/agents/greedy.py`
- `src/marlsim/agents/planning.py`

The agent layer is intentionally simple. Every policy implements
`choose_action(...)` and returns one `Action`.

Current policies:

- `RandomAgent` chooses uniformly from all actions.
- `GreedyGoalAgent` moves toward the goal using a Manhattan-distance heuristic.
- `ShortestPathAgent` uses BFS to choose the next step on a shortest path.

This is a good Phase 1 baseline because policies are transparent and easy to
debug.

### Planning

File:

- `src/marlsim/planning/bfs.py`

The planning package currently contains `shortest_path(...)`, a deterministic
breadth-first search over grid cells. BFS is a good fit for Phase 1 because all
movement costs are uniform and the algorithm is understandable for beginners.

### Scenarios, Rendering, And Demo

Files:

- `src/marlsim/scenarios/simple.py`
- `src/marlsim/visualization/console_renderer.py`
- `src/marlsim/demos/basic_grid.py`

The scenario layer provides reusable configurations. The renderer converts the
current environment into text. The demo wires together a scenario, environment,
renderer, and shortest-path policies.

This separation is good. It prevents demo code and visualization details from
leaking into the simulator rules.

### Tests

Files:

- `tests/test_environment.py`
- `tests/test_agents.py`
- `tests/test_planning.py`

The tests cover the important Phase 1 behaviors:

- normal movement
- out-of-bounds blocking
- obstacle blocking
- same-cell conflicts
- priority conflict policy
- direct swap blocking
- stationary-agent occupancy blocking
- goal termination
- BFS path planning
- greedy and shortest-path policy behavior

The current test suite is small but valuable. It also doubles as beginner
documentation for expected simulator behavior.

## Cleanup Needs

### 1. Keep `GridWorldEnv.step()` From Growing Too Large

`GridWorldEnv.step()` is currently understandable, but it is the main complexity
hotspot. It owns action normalization, proposed moves, invalid-move blocking,
multi-agent conflict resolution, state application, event generation, and
termination.

Recommended cleanup later:

- Extract small private helpers for each step phase.
- Keep helper names close to the simulator concepts, such as
  `_normalize_actions`, `_compute_requested_positions`, `_block_invalid_moves`,
  `_apply_final_positions`, and `_build_step_result`.
- Avoid adding learning, metrics, rendering, or scenario-specific logic inside
  `GridWorldEnv.step()`.

Priority: medium. The function is fine now, but it should not absorb future
features.

### 2. Replace Raw Blocked-Reason Strings Eventually

Blocked reasons are currently plain strings such as:

- `out_of_bounds`
- `obstacle`
- `swap_conflict`
- `occupied`
- `cell_conflict`

This is acceptable for Phase 1, but string literals become harder to maintain as
tests and docs grow.

Recommended cleanup later:

- Introduce a small `BlockedReason` enum.
- Use enum values in `AgentStepEvent`.
- Keep display strings near the enum instead of scattered across environment
  logic and tests.

Priority: low to medium. This is not urgent, but it will improve discoverability.

### 3. Consider A Read-Only Policy Context

`AgentPolicy.choose_action(...)` currently receives several separate arguments:
the current agent, all agents, obstacles, width, and height.

That is clear at the current size. If policies need more read-only information
later, the signature may become noisy.

Recommended cleanup later:

- Introduce a lightweight `PolicyContext` dataclass only when the policy
  signature starts to grow.
- Keep it read-only from the policy perspective.
- Do not pass the mutable environment object directly into policies.

Priority: low. The current signature is beginner-friendly and should not be
abstracted prematurely.

### 4. Tighten Milestone Language In Docs

`docs/project_plan.md` says Phase 2 is "started with shortest-path movement,"
while the current request is to complete Phase 1 professionally. The code does
include BFS and a shortest-path agent, but the milestone language could be more
precise.

Recommended cleanup later:

- Mark Phase 1 as implemented or under review once this review is accepted.
- Describe BFS as a Phase 1 baseline or Phase 2 seed, not a full Phase 2
  completion.
- Remove stale wording that says the next major step is adding BFS, because BFS
  already exists.

Priority: medium. Clear milestone wording matters for a portfolio repository.

### 5. Make Beginner Onboarding More Direct

The README includes run commands, but a new Python learner may not understand why
`PYTHONPATH=src` is needed before the package is installed.

Recommended cleanup later:

- Add a short "How the demo is wired" section.
- Explain that `PYTHONPATH=src` lets Python find the local package without
  installing it.
- Point readers from `basic_grid.py` to the scenario, policies, environment, and
  renderer.

Priority: medium. This improves readability without changing behavior.

### 6. Keep Public Imports Consistent

The top-level `marlsim/__init__.py` exports selected core symbols, while
subpackage `__init__.py` files are minimal. This is not a bug, but the public API
story should stay intentional.

Recommended cleanup later:

- Decide which imports are meant for users.
- Either keep the top-level exports small or add consistent exports in
  subpackages.
- Avoid changing import paths casually once examples and docs rely on them.

Priority: low.

## Improvement Suggestions

### Short-Term Improvements

These are safe follow-ups after Phase 1.1:

- Update `docs/project_plan.md` milestone statuses.
- Add a README walkthrough for the demo flow.
- Add tests for invalid configuration cases, such as duplicate agent IDs,
  obstacles outside the grid, goals on obstacles, and duplicate starting cells.
- Add tests for missing actions defaulting to `stay`.
- Add tests for invalid action strings raising a clear error.
- Add a test for `max_steps` termination.

### Medium-Term Improvements

These should wait until the current Phase 1 review is accepted:

- Split `GridWorldEnv.step()` into named internal phases if new behavior is
  added.
- Introduce `BlockedReason` if blocked reasons are referenced in more places.
- Add more deterministic coordination scenarios, such as a corridor, crossing
  paths, and a shared bottleneck.
- Add simple non-learning metrics such as steps taken, blocked moves, and goals
  reached.

### Explicitly Out Of Scope For Phase 1.1

Do not add these yet:

- reinforcement learning
- Gymnasium environment wrappers
- PyBullet
- ROS
- physics simulation
- neural network training
- complex graphics

Those directions depend on stable simulator behavior and should come after the
grid-world foundation is cleaned up, documented, and tested.

## Recommended Phase 1.1 Definition Of Done

Phase 1.1 should be considered complete when:

- The architecture is documented.
- Cleanup needs are listed without changing simulator behavior.
- Tests pass.
- The demo still runs.
- The next work item is a documentation/test cleanup task, not a new feature.

Verification performed during this review:

```powershell
$env:PYTHONPATH="src"; python -m unittest discover tests
```

Result:

```text
Ran 13 tests in 0.001s
OK
```

Demo verification:

```powershell
$env:PYTHONPATH="src"; python -m marlsim.demos.basic_grid
```

Result: both agents reached their goals after 6 steps.
