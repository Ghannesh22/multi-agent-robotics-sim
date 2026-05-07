# Phase 4.4 Training Metrics And Visualization

Phase 4.4 adds lightweight training-history reporting for tabular Q-learning.

The goal is visibility, not a dashboard. The reporting helpers stay dependency-free and produce plain text or CSV.

## Tracked Metrics

`TrainingMetrics` now includes:

- `episode_rewards`: total reward per episode.
- `episode_lengths`: number of steps per episode.
- `success_history`: whether each episode reached the goal.
- `timeout_history`: whether each episode timed out.
- `success_rate`: total successes divided by total episodes.
- `timeout_frequency`: total timeouts divided by total episodes.

These histories make it possible to inspect learning progress over time instead of only seeing final averages.

## Reporting Helpers

Phase 4.4 adds:

- `summarize_training(metrics)`: plain text training summary.
- `reward_chart(metrics)`: dependency-free text chart for episode rewards.
- `success_rate_chart(metrics)`: dependency-free rolling success-rate chart.
- `export_training_metrics(metrics, destination)`: CSV export for per-episode metrics.

CSV columns:

```text
episode,reward,length,success,timeout
```

## How To Interpret Learning Progress

Reward progression shows whether the agent is receiving better returns over time. In the current simple scenario, rewards should generally improve as the agent discovers goal-reaching actions.

Episode length shows whether the agent is reaching the goal in fewer steps. Shorter successful episodes are better in the current reward design because every step has a small penalty.

Success history shows whether the agent is reaching the goal more reliably.

Timeout history shows whether the agent is failing less often.

## Visualization Choice

Phase 4.4 uses ASCII/text charts instead of plotting libraries.

This keeps the project:

- Dependency-free.
- Easy to run in a terminal.
- Easy to test.
- Beginner-friendly.

Matplotlib or richer visualization can be considered later if the project needs saved plots or publication-quality figures.

## Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.q_learning_metrics_demo
```

The demo trains a small Q-learning agent, prints a summary, prints text charts, and shows a short CSV preview.

## Current Limitations

The current reporting does not include:

- Matplotlib plots.
- Web dashboards.
- Interactive charts.
- Multi-scenario aggregation.
- Hyperparameter comparisons.
- Saved experiment directories.

Those should wait until the simple metrics are useful and stable.
