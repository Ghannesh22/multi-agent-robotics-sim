# Phase 3.3 Action Design

Phase 3.3 keeps the single-agent RL wrapper's action interface simple and explicit.

## Action Format

`SingleAgentRLEnv.step(action)` accepts one integer action.

The action mapping is:

| Integer | Action name | Existing `Action` enum |
| ------- | ----------- | ---------------------- |
| `0` | `up` | `Action.UP` |
| `1` | `down` | `Action.DOWN` |
| `2` | `left` | `Action.LEFT` |
| `3` | `right` | `Action.RIGHT` |
| `4` | `stay` | `Action.STAY` |

The wrapper exposes this contract through:

- `action_names`
- `action_size`
- `action_to_env_action()`

## Why This Design Comes First

Five discrete actions are enough for the current grid-world simulator:

- Four movement directions.
- One explicit stay action.

This matches the existing `Action` enum and keeps the wrapper compatible with a future Gymnasium-style discrete action space without adding Gymnasium as a dependency in Phase 3.

## Invalid Actions

Invalid actions raise `ValueError`.

Invalid values include:

- Negative integers.
- Integers above `4`.
- Non-integer values.
- Boolean values, even though `bool` is a Python subclass of `int`.

Rejecting invalid actions early makes wrapper behavior easier to debug before any learning code exists.

## What It Does Not Include Yet

The Phase 3.3 action interface does not include:

- Continuous actions.
- Diagonal movement.
- Multi-action commands.
- Communication actions.
- Learned communication protocols.
- Multi-agent action dictionaries supplied by the learning algorithm.

Those features should wait until the single-agent grid-world learning loop is stable.
