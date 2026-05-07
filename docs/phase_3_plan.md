# Phase 3 Plan: Gymnasium-Style Reinforcement Learning Environment

## 1. Phase 3 Vision

Phase 3 makes the current simulator reinforcement-learning-ready without replacing the existing grid-world system.

In this project, RL-ready means the simulator has a stable interface that looks like a standard learning environment: reset an episode, submit one action, advance the world one step, observe the new state, receive a reward, and know whether the episode is over.

The Phase 3 wrapper should use the existing `GridWorldEnv`. It should not rewrite movement, collision prevention, obstacle handling, goal checks, scenarios, or step semantics. The wrapper's job is to translate the current simulator into an RL-friendly interface while keeping the core simulator understandable and testable.

## 2. Why Phase 3 Comes Before Learning

Before training any learning agent, the project needs a stable reset-step-observation-reward interface.

Training before this interface exists would make bugs hard to isolate. A failed training run could be caused by the reward function, action mapping, observation encoding, episode termination, scenario reset behavior, or the learning algorithm itself. Phase 3 separates those concerns by validating the environment contract first.

Phase 4 can start training safely only after Phase 3 proves that:

- `reset()` returns a valid initial observation.
- `step(action)` advances the existing environment consistently.
- Actions map clearly to simulator actions.
- Rewards are simple and documented.
- Termination and timeout behavior are predictable.
- Manual scripted action sequences behave correctly.

## 3. Phase 3 Sub-Phases

### Phase 3.0: RL Planning

Status: this document.

- Define RL interface.
- Decide observation representation.
- Decide action representation.
- Decide reward design.
- Decide episode lifecycle.
- Decide validation checks.
- No implementation yet.

### Phase 3.1: RL Environment Wrapper

- Create a wrapper around the existing `GridWorldEnv`.
- Provide `reset()`.
- Provide `step(action)`.
- Return observation, reward, done, and info.
- Do not change `GridWorldEnv` rules.

### Phase 3.2: Observation Design

- Define what the learning agent sees.
- Start with a simple vector observation.
- Include controlled agent position.
- Include controlled agent goal position.
- Include obstacle information only if simple.
- Avoid complex image or full-grid observations at the beginning.
- Phase 3.2 details are documented in `docs/phase_3_observation_design.md`.

### Phase 3.3: Action Design

- Use five discrete actions:
  - up
  - down
  - left
  - right
  - stay
- Map integer actions to the existing `Action` enum.
- Keep the action design compatible with later Gymnasium integration.
- Phase 3.3 details are documented in `docs/phase_3_action_design.md`.

### Phase 3.4: Reward Function

- Reward reaching the goal.
- Penalize each step slightly.
- Penalize blocked moves.
- Penalize timeout or failure.
- Keep reward values simple and documented.
- Phase 3.4 details are documented in `docs/phase_3_reward_design.md`.

### Phase 3.5: Episode Handling

- Define reset behavior.
- Define done or terminated condition.
- Define max-steps condition.
- Return a useful info dictionary.
- Keep single-agent training as the first target.
- Phase 3.5 details are documented in `docs/phase_3_episode_handling.md`.

### Phase 3.6: Manual RL-Environment Validation

- Test reset output.
- Test step output.
- Test reward values.
- Test done behavior.
- Test blocked-move penalty.
- Run manual scripted actions before training.
- Phase 3.6 details are documented in `docs/phase_3_manual_validation.md`.

### Phase 3.7: Phase 3 Finalization

- Add `docs/phase_3_summary.md`.
- Update README.
- Run all tests.
- Confirm environment is ready for Phase 4 training.
- Prepare final Phase 3 commit and tag.

## 4. Single-Agent First Strategy

Phase 3 and Phase 4 should start with one learning-controlled agent.

The current project is multi-agent, but multi-agent learning adds several hard problems at once: non-stationary behavior, credit assignment, coordination incentives, collision penalties, and policy interaction. Those problems should wait until the single-agent interface is stable.

The first RL wrapper should control one selected agent. If other agents are present, they should be static, scripted, or rule-based. They can be useful as moving obstacles or coordination baselines, but they should not learn during the first RL milestone.

The intended progression is:

- Phase 3: stable single-agent RL wrapper.
- Phase 4: train one learning-controlled agent.
- Later phases: add multi-agent learning only after the single-agent baseline is reliable.

## 5. Initial RL Environment Proposal

The proposed design is a lightweight wrapper class that owns or receives a `GridWorldEnv` instance and exposes a Gymnasium-style interface without requiring Gymnasium as a dependency.

Suggested future files:

- `src/marlsim/rl/__init__.py`
- `src/marlsim/rl/single_agent_env.py`
- `tests/test_rl_env.py`
- `docs/phase_3_summary.md`

The wrapper can be named something like `SingleAgentRLEnv`. It should keep configuration explicit:

- Scenario builder or environment factory.
- Controlled agent id.
- Optional policies for non-learning agents.
- Maximum episode steps.
- Reward values.

The wrapper should remain small. Its responsibilities are interface translation, reward calculation, episode bookkeeping, and validation. Core movement and collision rules should remain inside `GridWorldEnv`.

## 6. Proposed reset() Behavior

`reset()` should:

- Create or reset the configured scenario.
- Choose or validate the controlled agent.
- Reset episode counters.
- Reset terminal state.
- Return the initial observation.

The simplest implementation should recreate the scenario from a builder or factory function. That avoids depending on hidden mutable state from a previous episode.

The controlled agent should be selected by a stable id, such as `agent_1`. If the selected agent does not exist in the scenario, reset should fail clearly instead of silently selecting another agent.

## 7. Proposed step(action) Behavior

`step(action)` should:

- Validate the integer action.
- Map the integer action to the existing `Action` enum.
- Combine the controlled agent's action with fixed or rule-based actions for other agents if needed.
- Call `GridWorldEnv.step()`.
- Compute the reward from the resulting transition.
- Update episode counters.
- Determine whether the episode is done.
- Return observation, reward, done, and info.

The first version can use `done` as a single boolean. A later Gymnasium-compatible version may split this into `terminated` and `truncated`, but Phase 3 should not add that complexity unless it is needed.

The info dictionary should expose useful debugging data without becoming a second observation channel. Useful fields may include:

- `controlled_agent_id`
- `step_count`
- `reached_goal`
- `blocked`
- `blocked_reason`
- `timeout`

## 8. Observation Design Options

### Coordinate Vector

A coordinate vector stores a small numeric state:

- `agent_x`: controlled agent x coordinate, where x is the grid column.
- `agent_y`: controlled agent y coordinate, where y is the grid row.
- `goal_x`: controlled agent goal x coordinate.
- `goal_y`: controlled agent goal y coordinate.

This is the recommended starting point because it is easy to inspect, test, and feed into a simple learning algorithm later.

### Flattened Grid

A flattened grid encodes every cell as a number and returns the full grid as one vector.

This gives the agent more spatial context, including obstacles and other agents, but it also increases the observation size and requires careful encoding choices. It is useful later, but not necessary for the first RL wrapper.

### Dictionary-Style Observation

A dictionary observation can separate fields, for example:

- `agent_position`
- `goal_position`
- `obstacles`
- `other_agents`

This is expressive and readable, but it is more complex to validate and less convenient for simple baseline learning code unless the training library supports dictionary observations cleanly.

### Recommendation

Start with the coordinate vector.

It is the smallest representation that supports the first learning task: move one controlled agent from its current position to its goal in the existing grid world. Add obstacles or full-grid observations only after the coordinate-vector interface is tested.

## 9. Reward Design Proposal

Use simple starting values:

- `+10` for reaching the goal.
- `-0.1` per step.
- `-1` for a blocked move.
- `-5` for timeout or failure.

These values should be documented constants, not hidden magic numbers.

The first reward function should be intentionally simple. It should encourage reaching the goal quickly, discourage repeatedly invalid moves, and end failed episodes with a clear penalty. More complex reward shaping, such as distance-to-goal deltas or coordination-specific rewards, should wait until the basic wrapper is validated.

## 10. Testing Plan

Phase 3 implementation should add focused tests for the wrapper:

- `reset()` returns a valid observation.
- `step(action)` returns observation, reward, done, and info.
- Reaching the goal gives a positive reward.
- A blocked move gives the blocked-move penalty.
- Timeout ends the episode.
- The wrapper does not change `GridWorldEnv` behavior.

The behavior-preservation test is important. It should verify that the same action passed through the wrapper produces the same movement result as calling `GridWorldEnv.step()` directly for the controlled agent, aside from wrapper-only reward and episode bookkeeping.

## 11. What NOT To Do In Phase 3

Phase 3 should not add:

- RL training.
- Neural networks.
- Stable-Baselines3.
- Gymnasium dependency unless explicitly approved later.
- PyBullet.
- ROS.
- Physics simulation.
- Multi-agent RL.
- Communication learning.
- Complex reward shaping.
- Image observations.

These exclusions keep Phase 3 focused on the environment interface that Phase 4 will train against.

## 12. Success Criteria

Phase 3 should be considered successful when:

- The RL wrapper works.
- The reset/step interface is stable.
- Observation, action, and reward behavior are documented.
- Tests pass.
- Manual scripted runs behave correctly.
- Phase 4 can start training safely.

## 13. Risk Control

The main risk in Phase 3 is overengineering the RL layer before a simple learning problem works.

Risk controls:

- One controlled agent first.
- Simple observations first.
- Simple rewards first.
- No training until the interface is tested.
- No physics until grid-world learning works.
- Reuse `GridWorldEnv` instead of rewriting simulator rules.
- Keep Gymnasium compatibility in mind without adding the dependency yet.
- Add only the tests needed to lock the wrapper contract.

## 14. Next Implementation Step

The next step after Phase 3.0 is Phase 3.1: RL environment wrapper skeleton.

Phase 3.1 should create the smallest useful wrapper around `GridWorldEnv`, expose `reset()` and `step(action)`, map five integer actions to the existing `Action` enum, return observation, reward, done, and info, and add focused tests. It should not add training code, neural networks, Gymnasium, PyBullet, ROS, or changes to core environment rules.
