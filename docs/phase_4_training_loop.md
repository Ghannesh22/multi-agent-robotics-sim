# Phase 4.2 Q-Learning Training Loop

Phase 4.2 adds the first simple training loop for the tabular Q-learning agent.

The training loop uses the existing `SingleAgentRLEnv` and `QLearningAgent`. It does not add neural networks, deep RL, plotting, Gymnasium, Stable-Baselines3, PyTorch, TensorFlow, or multi-agent learning.

## How The Training Loop Works

`train_q_learning(env, agent, episodes)` runs a fixed number of episodes.

For each episode, it:

- Calls `env.reset()`.
- Uses the reset observation as the initial state.
- Chooses an integer action with `agent.choose_action(state)`.
- Calls `env.step(action)`.
- Updates the Q-table with `agent.update(state, action, reward, next_state, done)`.
- Accumulates episode reward.
- Counts episode length.
- Stops when the wrapper returns `done=True`.

The loop returns `TrainingMetrics`.

## Training Metrics

The returned metrics include:

- `total_episodes`: number of episodes requested.
- `success_count`: episodes where the controlled agent reached its goal.
- `failure_count`: episodes that did not reach the goal.
- `timeout_count`: episodes that ended by timeout.
- `total_reward`: sum of all episode rewards.
- `total_steps`: sum of all episode lengths.
- `episode_rewards`: reward total for each episode.
- `episode_lengths`: step count for each episode.
- `average_reward`: `total_reward / total_episodes`.
- `average_episode_length`: `total_steps / total_episodes`.

These metrics are intentionally simple. They are enough to check whether training is running and whether learning can later be compared with baseline policies.

## Why Exploration Matters

`QLearningAgent` uses epsilon-greedy action selection.

Exploration matters because an untrained Q-table starts with all actions valued at `0.0`. If the agent only exploits the current best value, it may repeat early actions and learn slowly. Epsilon-greedy exploration lets the agent occasionally try different actions, discover goal-reaching behavior, and update more useful Q-values.

## Current Limitations

Phase 4.2 is only the first training loop. It does not include:

- Baseline comparison.
- Advanced evaluation metrics.
- Plots.
- Hyperparameter sweeps.
- Scenario suites.
- Saved Q-tables.
- Multi-agent learning.
- Deep reinforcement learning.

Those should be added later after the simple training loop is stable and tested.

## Demo

Run the lightweight demo with:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_demo
```

The demo trains on a tiny single-agent scenario, prints a training summary, and prints learned Q-values for the start state.
