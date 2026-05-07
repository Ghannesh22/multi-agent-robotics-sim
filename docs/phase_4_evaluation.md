# Phase 4.3 Evaluation And Comparison

Phase 4.3 adds simple policy evaluation for the single-agent RL setup.

Evaluation is separate from training. It runs fixed episodes, collects metrics, and does not update learned Q-values.

## How Evaluation Differs From Training

Training changes the Q-table. During training, the agent explores, receives rewards, and updates Q-values after each step.

Evaluation measures policy behavior after training or for a fixed baseline. During evaluation:

- Episodes still call `reset()` and `step(action)`.
- Rewards and episode lengths are recorded.
- Q-values are not updated.
- Baseline policies are run exactly as policies.
- Q-learning exploration is disabled.

This makes policy comparisons easier to interpret.

## Why Exploration Is Disabled During Evaluation

The Q-learning agent uses epsilon-greedy exploration during training. Exploration is useful while learning because it lets the agent try actions it does not currently believe are best.

During evaluation, exploration should be disabled because the goal is to measure the learned policy, not random exploratory behavior. `evaluate_policy(...)` temporarily sets a `QLearningAgent`'s `epsilon` to `0.0` during evaluation and restores the original value afterward.

## Metrics

Evaluation reports:

- `success_rate`: fraction of episodes where the controlled agent reached its goal.
- `average_reward`: mean episode reward.
- `average_episode_length`: mean number of steps per episode.
- `timeout_frequency`: fraction of episodes that ended by timeout.

These metrics are intentionally small and reproducible. They are enough to compare the first learned policy against existing baselines.

## Compared Policies

Phase 4.3 compares:

- `RandomAgent`
- `GreedyGoalAgent`
- `ShortestPathAgent`
- trained `QLearningAgent`

## Policy Strengths And Weaknesses

`RandomAgent` is useful as a lower baseline. It explores without learning, so it can succeed by chance but is inefficient and may time out.

`GreedyGoalAgent` is simple and deterministic. It moves toward the goal using local information, but it can fail in scenarios where a locally good move is not globally good.

`ShortestPathAgent` is strong in simple grid worlds because it uses BFS planning. It is a useful non-learning upper baseline for the first Q-learning experiments.

`QLearningAgent` learns from rewards and can improve over random behavior. In Phase 4.3 it is still tabular and scenario-limited, so it should not be expected to beat shortest-path planning in simple deterministic layouts.

## Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_comparison
```

The demo trains a Q-learning agent on a tiny single-agent scenario, evaluates all policies, and prints a text comparison table.
