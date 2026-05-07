# Phase 3.5 Episode Handling

Phase 3.5 makes the single-agent RL wrapper episode lifecycle explicit and testable.

## Episode State

`SingleAgentRLEnv` tracks:

- `step_count`: current underlying `GridWorldEnv` step count.
- `done`: true when the wrapper episode is over.
- `episode_done`: read-only alias for `done`.
- `reached_goal`: true when the controlled agent reaches its goal.
- `timeout`: true when max steps are reached before the controlled agent reaches its goal.
- `max_steps`: the max-step limit from the wrapped `GridWorldEnv` config.

The wrapper episode ends when:

- The controlled agent reaches its goal.
- The wrapped `GridWorldEnv` reaches `max_steps` before the controlled agent reaches its goal.

## reset()

`reset()` resets:

- The underlying `GridWorldEnv`.
- `step_count` to `0`.
- `done` to `False`.
- `reached_goal` to `False`.
- `timeout` to `False`.

It then returns the initial observation:

```text
(agent_x, agent_y, goal_x, goal_y)
```

## step(action)

`step(action)`:

- Rejects calls after the episode is done.
- Maps the integer action to the existing `Action` enum.
- Applies the controlled agent action.
- Applies `stay` for other agents for now.
- Calls `GridWorldEnv.step()`.
- Updates episode state.
- Computes reward.
- Returns observation, reward, done, and info.

## Step After Done

Calling `step()` after the wrapper episode is done raises:

```text
RuntimeError: Cannot call step() after the episode is done. Call reset() first.
```

This keeps episode boundaries clear before training code exists.

## Info Dictionary

Every successful `step()` returns an info dictionary with:

- `controlled_agent_id`
- `action`
- `step_count`
- `reached_goal`
- `blocked`
- `blocked_reason`
- `timeout`
- `done`

The info dictionary is for debugging and validation. It should not become a hidden observation channel for future learning code.

## What Is Intentionally Not Included Yet

Phase 3.5 does not add:

- Gymnasium `terminated` and `truncated` split.
- Auto-reset behavior.
- Multi-agent episode termination rules.
- Training loops.
- Vectorized environments.
- Curriculum logic.

Those can wait until the single-agent wrapper contract is stable.
