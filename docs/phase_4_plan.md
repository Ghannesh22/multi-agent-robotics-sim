# Phase 4 Plan: Single-Agent Reinforcement Learning Baseline

## 1. Phase 4 Vision

Phase 4 is the project's first real learning phase.

The goal is to train one learning-controlled agent in the existing grid-world using the Phase 3 `SingleAgentRLEnv` wrapper. Phase 4 should stay single-agent first, simple, and inspectable. Other agents, if used later, should remain fixed or rule-based until the first learned policy is stable.

This phase should use simple reinforcement learning before deep reinforcement learning. The first learned baseline should be tabular Q-learning, not a neural network policy. That makes the learning behavior easier to understand, debug, and compare against the existing random, greedy, shortest-path, and rule-based coordination baselines.

Phase 4 should answer a narrow question:

Can a simple tabular learning agent improve over random behavior in the current RL-ready grid world?

## 2. Why Tabular Q-Learning First

Tabular Q-learning is the right first training method because the current environment is discrete and small.

The Phase 3 wrapper already exposes:

- A tuple observation: `(agent_x, agent_y, goal_x, goal_y)`.
- Five discrete actions: up, down, left, right, stay.
- A simple reward function.
- Clear done and timeout behavior.

That is enough for a Q-table. Each discrete observation can be used directly as a state key, and each state-action pair can store one numeric Q-value.

Q-learning is useful first because it is:

- Easy to debug with printed Q-values.
- Easy to test without training frameworks.
- Easy to explain and reason about.
- Small enough for the current grid-world scenarios.
- A correct stepping stone before neural networks.

Deep RL should wait until the project has a known-good training loop, metrics, and evaluation process.

## 3. Phase 4 Sub-Phases

### Phase 4.0: RL Training Planning

Status: this document.

- Define training strategy.
- Define evaluation metrics.
- Define comparison approach.
- Define risks.
- No implementation.

### Phase 4.1: Q-Table Agent

- Create a Q-table structure.
- Implement epsilon-greedy action selection.
- Implement Q-value update rule.
- Keep state keys as tuples.
- Keep action ids as integers.
- No deep learning.
- Phase 4.1 details are documented in `docs/phase_4_q_learning.md`.

### Phase 4.2: Training Loop

- Run episodes.
- Reset the RL wrapper at the start of each episode.
- Select actions with the Q-table agent.
- Collect rewards.
- Update the Q-table after each step.
- Track success rate.
- Track episode length.
- Track total episode reward.
- Phase 4.2 details are documented in `docs/phase_4_training_loop.md`.

### Phase 4.3: Evaluation And Comparison

- Compare random policy.
- Compare greedy policy.
- Compare shortest-path baseline.
- Compare trained Q-learning agent.
- Compare steps and success rate.
- Keep evaluation deterministic where possible.
- Phase 4.3 details are documented in `docs/phase_4_evaluation.md`.

### Phase 4.4: Training Metrics And Visualization

- Track reward curves.
- Track episode-length curves.
- Track success-rate curves.
- Use simple plots only.
- Prefer reproducible text summaries before adding richer visualization.
- Phase 4.4 details are documented in `docs/phase_4_metrics.md`.

### Phase 4.5: Final Phase 4 Polish

- Update documentation.
- Add final tests.
- Add result summary.
- Run all tests.
- Run training and evaluation demos.
- Prepare final Phase 4 commit and tag.

## 4. Proposed RL Structure

Suggested future files:

- `src/marlsim/rl/q_learning.py`
- `src/marlsim/rl/training.py`
- `src/marlsim/demos/q_learning_demo.py`
- `tests/test_q_learning.py`
- `docs/phase_4_summary.md`

The implementation should stay small. Phase 4 should avoid a large training framework, plugin system, experiment database, or neural-network abstraction.

## 5. Initial Q-Learning Design

### State Representation

Use the current Phase 3 observation tuple directly:

```text
(agent_x, agent_y, goal_x, goal_y)
```

This tuple should be the Q-table state key.

### Action Representation

Use the current Phase 3 integer action ids:

| Integer | Action |
| ------- | ------ |
| `0` | up |
| `1` | down |
| `2` | left |
| `3` | right |
| `4` | stay |

### Q-Table Structure

Use a mapping from state to action values:

```text
Q[state][action] -> float
```

For a first implementation, the table can be a dictionary with tuple state keys and lists or dictionaries of action values.

### Q-Value Update Rule

Use the standard Q-learning update:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

If the transition ends the episode, the future-value term should be `0`.

### Simple Defaults

Recommended starting defaults:

- `alpha = 0.1`
- `gamma = 0.95`
- `epsilon = 0.1`

These are starting values only. They should be documented and easy to change.

### Epsilon-Greedy Exploration

The Q-table agent should:

- Choose a random action with probability `epsilon`.
- Choose the highest-Q action otherwise.
- Break ties deterministically or with a clearly documented random choice.

For reproducibility, the implementation should accept a random seed or random number generator.

## 6. State Representation Strategy

Start with tuple-based discrete state.

The current observation vector should be used directly. This keeps the implementation aligned with Phase 3 and avoids adding a second state encoding layer before it is needed.

Phase 4 should avoid:

- Neural embeddings.
- Image observations.
- Flattened grid observations unless a later experiment needs them.
- Hand-built feature vectors beyond the existing coordinate tuple.
- Hidden state or recurrent memory.

The first learning target should prove that the reset-step-reward loop works.

## 7. Evaluation Plan

Evaluate policies using:

- Success rate.
- Average steps.
- Average reward.
- Timeout frequency.

Initial comparisons should include:

- Random policy.
- Greedy policy.
- Shortest-path baseline.
- Trained Q-learning agent.

The first evaluation should run on the simplest scenario. Later checks can include more scenarios after the basic learning loop works.

Evaluation should report enough information to answer:

- Did the Q-learning agent reach the goal more often than random?
- Did it use fewer steps than random?
- Did it time out less often?
- Did training remain reproducible across runs with the same seed?

## 8. Training Safety Rules

Phase 4 should follow these safety rules:

- Start with the simplest scenario first.
- Use one controlled learning agent only.
- Keep any other agents static or rule-based.
- Do not start multi-agent learning yet.
- Do not add deep RL yet.
- Do not add neural networks yet.
- Keep training runs short enough to inspect.
- Keep metrics text-based before adding plots.

## 9. What NOT To Do In Phase 4

Phase 4 should not add:

- DQN.
- PPO.
- A2C.
- Transformers.
- PyTorch.
- TensorFlow.
- Stable-Baselines3.
- Multi-agent RL.
- Communication learning.
- PyBullet.
- ROS.
- Physics simulation.

These tools and methods can be considered later, after a simple tabular baseline is implemented, tested, and understood.

## 10. Success Criteria

Phase 4 should be considered successful when:

- A Q-learning agent trains successfully.
- Learning improves over random behavior.
- Metrics are reproducible.
- Training remains understandable.
- The project stays beginner-friendly but technically solid.
- Existing Phase 1, Phase 2, and Phase 3 tests still pass.
- No deep learning framework is needed.

## 11. Risk Control

The main Phase 4 risks are unstable training, overcomplicated infrastructure, and premature deep learning.

Risk controls:

- Keep rewards unchanged at first.
- Use the existing Phase 3 wrapper exactly as tested.
- Start with one small scenario.
- Keep the state space limited to the coordinate tuple.
- Add Q-learning before adding plotting.
- Add plotting only after metrics are correct.
- Use seeds for reproducibility.
- Compare against simple baselines before tuning.
- Do not introduce neural networks until tabular learning is working.

Avoid unstable rewards by treating Phase 3 reward values as the first baseline. If rewards need tuning, record the reason and compare before and after.

Avoid huge state spaces by not adding full grids, image observations, or multi-agent state until the single-agent coordinate-state baseline works.

Avoid overcomplicated training by keeping the first training loop in a small module with explicit episode and step loops.

## 12. Next Implementation Step

The next step after Phase 4.0 is Phase 4.1: Q-table agent implementation.

Phase 4.1 should create the smallest useful Q-learning agent:

- Q-table storage.
- Epsilon-greedy action selection.
- Q-value update.
- Focused unit tests.

It should not add neural networks, PyTorch, TensorFlow, Stable-Baselines3, Gymnasium, DQN, PPO, A2C, multi-agent RL, PyBullet, ROS, or physics simulation.
