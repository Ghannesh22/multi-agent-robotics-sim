# Phase 4 Summary

Phase 4 completed the project's first reinforcement learning milestone: a simple, tabular, single-agent Q-learning baseline.

Phase 4 built directly on the Phase 3 `SingleAgentRLEnv` wrapper. It did not change the core `GridWorldEnv` rules, the RL observation contract, the RL action contract, or the reward values.

## What Phase 4 Achieved

- Created a Phase 4 plan in `docs/phase_4_plan.md`.
- Added `QLearningAgent`.
- Added tabular Q-table storage.
- Added epsilon-greedy action selection.
- Added the Q-learning update rule.
- Added a simple training loop.
- Added evaluation against baseline policies.
- Added training-history metrics.
- Added dependency-free text reporting and ASCII-style charts.
- Added CSV export for training metrics.
- Added lightweight demos for training, comparison, and metrics.
- Added focused tests for learning, training, evaluation, and reporting.

## How Tabular Q-Learning Works In This Project

The Q-learning agent uses the Phase 3 observation tuple directly as the state:

```text
(agent_x, agent_y, goal_x, goal_y)
```

The action space is the existing Phase 3 integer action interface:

| Integer | Action |
| ------- | ------ |
| `0` | up |
| `1` | down |
| `2` | left |
| `3` | right |
| `4` | stay |

The Q-table stores one value for each state-action pair:

```text
q_table[state][action] -> float
```

The update rule is:

```text
Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

The initial defaults are:

- `learning_rate = 0.1`
- `discount_factor = 0.95`
- `epsilon = 0.1`

Unseen Q-values default to `0.0`.

## Training

`train_q_learning(...)` runs repeated episodes through `SingleAgentRLEnv`.

For each episode, the training loop:

- Calls `reset()`.
- Chooses integer actions with epsilon-greedy exploration.
- Calls `step(action)`.
- Updates the Q-table.
- Accumulates reward.
- Tracks episode length.
- Tracks whether the episode reached the goal or timed out.

The training loop returns `TrainingMetrics`.

## Evaluation And Comparison

Phase 4 added `evaluate_policy(...)` for fixed-policy evaluation without learning updates.

Evaluation compares:

- `RandomAgent`
- `GreedyGoalAgent`
- `ShortestPathAgent`
- trained `QLearningAgent`

Evaluation metrics:

- Success rate.
- Average reward.
- Average episode length.
- Timeout frequency.

During evaluation, Q-learning exploration is disabled by temporarily setting epsilon to `0.0` and restoring it afterward.

In the current tiny deterministic scenario, the trained Q-learning agent improves over random behavior and can match the greedy and shortest-path baselines after a short training run.

## Metrics And Reporting

Phase 4 tracks:

- `episode_rewards`
- `episode_lengths`
- `success_history`
- `timeout_history`
- aggregate success rate
- aggregate timeout frequency
- average reward
- average episode length

Reporting helpers include:

- `summarize_training(...)`
- `reward_chart(...)`
- `success_rate_chart(...)`
- `export_training_metrics(...)`

The visualization is dependency-free text output. Phase 4 intentionally does not add Matplotlib, dashboards, or web apps.

## Demos

Run basic Q-learning training:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_demo
```

Run policy comparison:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_comparison
```

Run metrics reporting:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_metrics_demo
```

## Strengths Of The Learned Policy

The learned Q-table is:

- Easy to inspect.
- Easy to test.
- Reproducible with seeds.
- Fast to train in tiny grid-world scenarios.
- Good enough to prove the reset-step-reward learning loop works.

The trained policy improves over random behavior in the current simple scenario.

## Weaknesses Of The Learned Policy

The current learned policy is limited:

- It is tabular, so it does not scale to large state spaces.
- It is trained on a tiny single-agent scenario.
- It does not yet evaluate across a broad scenario suite.
- It does not learn communication.
- It does not learn multi-agent coordination.
- It does not use function approximation.

It should be treated as a first baseline, not a final learning system.

## Why Deep RL Is Postponed

Deep RL is intentionally postponed because the project first needs trustworthy basics:

- stable environment interface
- clear tabular learning behavior
- reproducible training metrics
- baseline comparisons
- documentation and tests

Adding DQN, PPO, neural networks, PyTorch, TensorFlow, or Stable-Baselines3 before this point would make failures harder to debug.

## Intentionally Out Of Scope

Phase 4 intentionally did not add:

- DQN.
- PPO.
- A2C.
- Transformers.
- PyTorch.
- TensorFlow.
- Stable-Baselines3.
- Gymnasium dependency.
- Neural networks.
- Multi-agent RL.
- Communication learning.
- PyBullet.
- ROS.
- Physics simulation.
- Complex dashboards.

## Why Phase 5 Comes Next

Phase 4 proves that one learning-controlled agent can train and be evaluated in the grid-world.

Phase 5 can now explore multi-agent reinforcement learning from a stronger foundation. The next milestone should still proceed carefully: start with simple multi-agent learning questions, reuse the existing environment contracts, and avoid deep RL until tabular and single-agent baselines are well understood.
