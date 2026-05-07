# Phase 3 Summary

Phase 3 completed the project's first Gymnasium-style reinforcement learning environment milestone without adding training or changing the core grid-world simulator.

## What Phase 3 Achieved

- Created a Phase 3 planning roadmap in `docs/phase_3_plan.md`.
- Added the `marlsim.rl` package.
- Added `SingleAgentRLEnv`, a lightweight RL-style wrapper around `GridWorldEnv`.
- Added explicit observation, action, reward, and episode lifecycle contracts.
- Added focused tests for the wrapper interface.
- Added a manual validation demo in `src/marlsim/demos/rl_env_validation.py`.
- Preserved existing `GridWorldEnv` movement, collision, obstacle, goal, and max-step rules.

## RL Wrapper Behavior

`SingleAgentRLEnv` controls one selected agent, defaulting to `agent_1`.

For each call to `step(action)`, the wrapper:

- Validates one integer action.
- Maps the integer action to the existing `Action` enum.
- Applies that action to the controlled agent.
- Applies `Action.STAY` to other agents for now.
- Calls `GridWorldEnv.step()`.
- Computes reward from the transition.
- Updates episode state.
- Returns observation, reward, done, and info.

The wrapper is intentionally small. It translates the existing simulator into an RL-style interface, but it does not replace the simulator.

## Observation Format

The observation is a four-value coordinate vector:

```text
(agent_x, agent_y, goal_x, goal_y)
```

The values mean:

- `agent_x`: controlled agent x coordinate, where x is the grid column.
- `agent_y`: controlled agent y coordinate, where y is the grid row.
- `goal_x`: controlled agent goal x coordinate.
- `goal_y`: controlled agent goal y coordinate.

The wrapper exposes:

- `observation_names`
- `observation_size`
- `get_observation()`

This first observation design intentionally does not include obstacles, other agents, image observations, action history, reward history, or full-grid encodings.

## Action Mapping

The action space uses five discrete integer actions:

| Integer | Action name | Existing enum |
| ------- | ----------- | ------------- |
| `0` | `up` | `Action.UP` |
| `1` | `down` | `Action.DOWN` |
| `2` | `left` | `Action.LEFT` |
| `3` | `right` | `Action.RIGHT` |
| `4` | `stay` | `Action.STAY` |

The wrapper exposes:

- `action_names`
- `action_size`
- `action_to_env_action()`

Invalid integer actions raise a clear `ValueError`.

## Reward Function

The reward function is simple and additive:

| Component | Value | Applies when |
| --------- | ----- | ------------ |
| `step_penalty` | `-0.1` | Every step |
| `goal_reward` | `+10.0` | Controlled agent reaches its goal |
| `blocked_penalty` | `-1.0` | Controlled move is blocked |
| `timeout_penalty` | `-5.0` | Max steps are reached before the controlled agent reaches its goal |

The wrapper exposes:

- `step_penalty`
- `goal_reward`
- `blocked_penalty`
- `timeout_penalty`
- `reward_values`
- `compute_reward(...)`

Reward shaping is intentionally minimal. Phase 3 does not add distance-to-goal shaping, coordination rewards, communication rewards, learned reward models, or scenario-specific reward tuning.

## Episode Lifecycle

The wrapper tracks:

- `step_count`
- `done`
- `episode_done`
- `reached_goal`
- `timeout`
- `max_steps`

An episode ends when:

- The controlled agent reaches its goal.
- The environment reaches `max_steps` before the controlled agent reaches its goal.

`reset()` resets the underlying `GridWorldEnv`, step counter, done flag, reached-goal flag, and timeout flag.

Calling `step()` after done raises a clear `RuntimeError` asking the caller to call `reset()` first.

The info dictionary returned by `step()` includes:

- `controlled_agent_id`
- `action`
- `step_count`
- `reached_goal`
- `blocked`
- `blocked_reason`
- `timeout`
- `done`

## Manual Validation Results

The manual validation demo runs with:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.rl_env_validation
```

It validates:

- Reset output.
- Normal movement step.
- Blocked movement step.
- Goal-reaching episode.
- Timeout episode.

Expected example outcomes:

```text
normal movement reward: -0.1
blocked movement reward: -1.1
goal-reaching reward: 9.9
timeout reward: -5.1
```

The validation script is fixed-action interface testing. It is not RL training.

## Intentionally Out Of Scope

Phase 3 intentionally did not add:

- RL training.
- Gymnasium dependency.
- Stable-Baselines3.
- Neural networks.
- PyBullet.
- ROS.
- Physics simulation.
- Multi-agent RL.
- Communication learning.
- Complex reward shaping.
- Image observations.

These exclusions keep Phase 3 focused on a stable, understandable environment interface.

## Why Phase 4 Comes Next

Phase 3 now provides the reset-step-observation-reward interface that Phase 4 needs.

Phase 4 can start with one learning-controlled agent because the wrapper behavior is documented, tested, and manually validated. Training should begin only after this stable interface exists, so training failures can be debugged as learning problems instead of environment-contract problems.
