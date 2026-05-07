# Phase 3.4 Reward Design

Phase 3.4 keeps the first single-agent RL reward function simple, explicit, and independently testable.

## Reward Values

The current reward values are:

| Reward component | Value | When applied |
| ---------------- | ----- | ------------ |
| `step_penalty` | `-0.1` | Every call to `step()` |
| `goal_reward` | `+10.0` | When the controlled agent reaches its goal |
| `blocked_penalty` | `-1.0` | When the controlled agent's move is blocked by `GridWorldEnv` |
| `timeout_penalty` | `-5.0` | When the episode reaches max steps before the controlled agent reaches its goal |

The reward for a transition is additive:

```text
reward = step_penalty
       + blocked_penalty if the move was blocked
       + goal_reward if the goal was reached
       + timeout_penalty if the episode timed out before the goal
```

`goal_reward` and `timeout_penalty` are mutually exclusive in the current wrapper because timeout is only true when max steps are reached and the controlled agent has not reached its goal.

## Why Each Reward Exists

The step penalty encourages shorter paths and discourages unnecessary waiting.

The goal reward makes successful task completion clearly better than ordinary movement.

The blocked-move penalty discourages repeatedly trying illegal moves into walls, obstacles, occupied cells, or conflict states.

The timeout penalty gives failed episodes a clear negative outcome, which will be useful later when comparing training runs.

## What Is Intentionally Not Included Yet

The Phase 3.4 reward function does not include:

- Distance-to-goal shaping.
- Extra reward for reducing Manhattan distance.
- Extra penalty for increasing distance.
- Collision-specific reward categories beyond the existing blocked move signal.
- Multi-agent cooperation rewards.
- Communication rewards.
- Learned reward models.
- Scenario-specific reward tuning.

Those features can make learning easier later, but they also make early environment bugs harder to diagnose. The first reward function should stay small until reset, step, observation, action, and episode behavior are fully validated.

## Wrapper API

`SingleAgentRLEnv` exposes the reward contract through:

- `step_penalty`
- `goal_reward`
- `blocked_penalty`
- `timeout_penalty`
- `reward_values`
- `compute_reward(...)`

The source of truth remains `RewardConfig`, so tests and future experiments can create alternate reward configurations without changing `GridWorldEnv`.
