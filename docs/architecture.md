# Architecture

The project is organized around a small set of separable concerns.

## Core Simulation

The `marlsim.core` package owns deterministic simulation behavior:

- `Position` and `AgentState` define immutable state primitives.
- `Action` defines the five discrete grid actions.
- `GridWorldEnv` owns reset, stepping, collision prevention, and termination.

The environment does not know how an agent chooses actions. This keeps random, rule-based, and learned policies interchangeable.

## Agents

The `marlsim.agents` package owns policy behavior:

- `RandomAgent` is useful for smoke tests and future RL baseline comparisons.
- `GreedyGoalAgent` is the simplest rule-based policy.
- `ShortestPathAgent` uses breadth-first search to route around obstacles.

Each policy receives the current agent, all agents, obstacles, and grid dimensions, then returns one discrete action.

## Planning

The `marlsim.planning` package owns reusable path-planning algorithms.

The first implementation is breadth-first search because it is simple, deterministic, and correct for uniform-cost grid movement.

## Visualization

The `marlsim.visualization` package renders environment state without changing simulation behavior.

The first renderer is plain text. Pygame, Matplotlib, or browser-based visualization can be added later without rewriting the environment.

## Tests

The `tests` directory uses Python `unittest` so the first milestone has no third-party dependency requirement.

Current test focus:

- Movement into empty cells.
- Out-of-bounds blocking.
- Obstacle blocking.
- Same-cell conflict blocking.
- Priority conflict behavior.
- Direct swap blocking.
- Stationary-agent occupancy blocking.
- Goal termination.
- BFS path planning.
