# Phase 3.2 Observation Design

Phase 3.2 keeps the single-agent RL wrapper's observation intentionally small and explicit.

## Observation Format

`SingleAgentRLEnv` returns a coordinate-vector observation:

```text
(agent_x, agent_y, goal_x, goal_y)
```

The values mean:

- `agent_x`: controlled agent x coordinate, where x is the grid column.
- `agent_y`: controlled agent y coordinate, where y is the grid row.
- `goal_x`: controlled agent goal x coordinate.
- `goal_y`: controlled agent goal y coordinate.

The wrapper exposes this contract through:

- `observation_names`
- `observation_size`
- `get_observation()`

## Why This Design Comes First

The coordinate vector is the smallest useful state for the first learning task: one controlled agent moving toward its own goal in the existing grid world.

This design is useful first because it is:

- Easy to print and inspect.
- Easy to test exactly.
- Independent of any Gymnasium dependency.
- Simple enough for early scripted validation.
- Small enough to avoid hiding environment-interface bugs behind observation complexity.

## What It Does Not Include Yet

The Phase 3.2 observation does not include:

- Obstacles.
- Other agents.
- Full grid state.
- Flattened grid encodings.
- Image observations.
- Action history.
- Reward history.
- Communication messages.
- Learned features.

These can be added later after the reset/step interface and simple observation behavior are stable.

## Example

If the controlled agent starts at `(0, 0)` and its goal is `(2, 0)`, reset returns:

```text
(0, 0, 2, 0)
```

After taking action `3` (`right`), the wrapper may return:

```text
(1, 0, 2, 0)
```

The goal coordinates stay fixed while the controlled agent coordinates update according to the existing `GridWorldEnv` step rules.
