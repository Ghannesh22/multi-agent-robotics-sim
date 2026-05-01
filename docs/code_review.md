# Code Review: v0.1.0

The current project state is frozen as `v0.1.0`. This review documents what exists now and what should come next without adding new simulation behavior.

## 1. What Currently Works

- The project has a clean Python package structure under `src/marlsim`.
- The grid-world environment supports fixed width and height, obstacles, agents, per-agent goals, and a maximum step count.
- Agents can choose from five discrete actions: up, down, left, right, and stay.
- The environment supports simultaneous multi-agent steps.
- Collision prevention is implemented for out-of-bounds moves, obstacle moves, same-cell conflicts, direct swaps, and moves into stationary agents.
- A console renderer displays obstacles, goals, and agents as text.
- Rule-based baseline agents exist:
  - `RandomAgent` chooses random actions.
  - `GreedyGoalAgent` moves toward the goal using simple Manhattan-distance logic.
  - `ShortestPathAgent` uses breadth-first search to choose the next step on a shortest path.
- The demo scenario runs successfully and reaches both goals.
- Unit tests cover the main movement, collision, goal termination, agent, and planning behaviors.

## 2. How The Environment Step Function Works

`GridWorldEnv.step()` processes one simultaneous environment update.

The function works in this order:

1. Capture each agent's starting position.
2. Normalize the requested action for every agent. Agents without an explicit action default to `stay`.
3. Compute each agent's requested target cell from its action delta.
4. Immediately block simple invalid moves:
   - Target is outside the grid.
   - Target is an obstacle.
5. Run multi-agent collision checks:
   - Block direct position swaps.
   - Block moves into agents that are staying in place.
   - Resolve same-cell conflicts.
   - Re-check moves into stationary agents after conflict resolution.
6. Replace the environment's agent states with the final positions.
7. Increment `step_count`.
8. Build `AgentStepEvent` records describing each agent's action, requested target, final position, movement result, and blocked reason.
9. Compute which agents reached their goals.
10. Terminate the episode if every agent has reached a goal or `max_steps` has been reached.

The return value is a `StepResult` containing the new agent states, per-agent events, reached goals, step count, and termination flag.

## 3. How Collision Prevention Works

Collision prevention is deterministic and conservative.

- Out-of-bounds moves are blocked before multi-agent resolution.
- Obstacle moves are blocked before multi-agent resolution.
- Direct swaps are blocked when two agents try to move into each other's starting cells in the same step.
- Same-cell conflicts are grouped by final target cell:
  - With `BLOCK_ALL`, every agent trying to enter the same target cell is blocked.
  - With `PRIORITY`, the lexicographically first agent ID wins and the others are blocked.
- Moves into stationary agents are blocked. This also handles cascades where one blocked agent causes another agent's target to become occupied.

The current behavior favors safety and clarity over complex negotiation. That is appropriate for v0.1.0 because later learning and coordination work needs predictable environment rules.

## 4. How The Demo Reaches The Goals

The demo uses `two_agent_obstacle_course()`:

- Grid size: 7 columns by 5 rows.
- Obstacles form a vertical wall in column 3, rows 1 through 3.
- `agent_1` starts at `(0, 0)` and has goal `(6, 0)`.
- `agent_2` starts at `(0, 4)` and has goal `(6, 4)`.
- Both agents use `ShortestPathAgent`.

Because the top and bottom rows are open, each agent's BFS path is a straight horizontal route to its goal:

- `agent_1` moves right along the top row.
- `agent_2` moves right along the bottom row.
- They do not compete for the same cells.
- They reach their goals after 6 steps.

The demo is intentionally simple. It confirms that the environment, agent policy, BFS planner, scenario config, and console renderer work together.

## 5. Next Safe Milestone

The next safe milestone should be a Phase 2 hardening milestone, not reinforcement learning yet.

Recommended milestone:

**Phase 2.1: Coordination Scenarios And Rule-Based Baselines**

Scope:

- Add more deterministic scenarios that create real coordination pressure, such as narrow corridors, crossing paths, and shared bottlenecks.
- Expand tests around conflict resolution, blocked paths, and priority behavior.
- Add simple metrics for completed goals, blocked moves, collisions prevented, and total steps.
- Keep agents rule-based and transparent.
- Do not add reinforcement learning until the environment behavior is stable across several scenarios.

Why this is the safest next step:

- Reinforcement learning depends heavily on stable environment semantics.
- More coordination scenarios will reveal edge cases before training begins.
- Basic metrics will make later learned-vs-rule-based comparisons meaningful.
- The project remains simple and understandable while moving toward the original roadmap.
