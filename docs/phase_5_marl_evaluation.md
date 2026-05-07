# Phase 5.4 MARL Evaluation and Metrics

Phase 5.4 adds an evaluation layer for multi-agent reinforcement learning behavior.

Evaluation is separate from training. Training updates Q-tables while agents explore. Evaluation runs fixed policies without learning updates so results are easier to compare.

## What Is Evaluated

The MARL evaluator supports:

- Multiple independent `QLearningAgent` policies.
- Random baseline policies.
- Greedy and shortest-path baseline policies.
- Rule-based coordination policies from Phase 2.

The evaluator uses the existing `MultiAgentRLEnv` and does not change `GridWorldEnv` rules.

## Why Exploration Is Disabled

During training, Q-learning agents use epsilon-greedy exploration. Exploration is useful for discovering behavior, but it makes evaluation noisy.

During evaluation:

- Each Q-learning agent's `epsilon` is temporarily set to `0.0`.
- Greedy Q-table actions are selected.
- The original epsilon values are restored afterward.
- Q-tables are not updated.

This makes evaluation closer to asking: "What has the policy learned?"

## Metrics

`MARLEvaluationMetrics` reports:

- `success_rate`: fraction of episodes where all controlled agents reached goals.
- `timeout_frequency`: fraction of episodes that ended by timeout.
- `blocked_move_count`: total blocked moves across controlled agents.
- `average_reward`: average combined reward per episode.
- `average_episode_length`: average number of environment steps per episode.
- `final_positions`: final controlled-agent positions for each episode.

These metrics intentionally stay simple and reproducible.

## Comparison Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_evaluation_comparison
```

The demo:

- Trains independent Q-learning agents with individual rewards.
- Trains independent Q-learning agents with shared rewards.
- Evaluates both learned policies with exploration disabled.
- Compares them against random, greedy, shortest-path, and rule-based coordination baselines.
- Prints a compact text table.

## Current Results

Results depend on the small scenario and random seeds used in the demo. The current evaluation is meant to verify the workflow, not to claim a final MARL benchmark.

The important behavior is:

- Training and evaluation are separate.
- Learned Q-tables are not changed during evaluation.
- Baselines and learned policies use the same environment and metrics.
- The comparison table is reproducible enough for Phase 5 development.

## Current Limitations

Phase 5.4 does not add:

- Deep RL.
- Neural networks.
- Centralized critics.
- Parameter sharing.
- Communication protocols.
- RLlib or PettingZoo.
- Stable-Baselines3.
- PyTorch or TensorFlow.
- Physics simulation.
- ROS.

The goal is a clean evaluation layer for simple independent tabular MARL.
