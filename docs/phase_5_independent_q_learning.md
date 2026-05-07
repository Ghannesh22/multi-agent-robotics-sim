# Phase 5.2 Independent Multi-Agent Q-Learning

Phase 5.2 adds the first multi-agent learning loop.

The approach is independent Q-learning: each controlled agent owns its own `QLearningAgent` and Q-table. Agents act in the same environment step, then each learner updates independently from its own observation, action, reward, next observation, and done flag.

## What Independent Q-Learning Means

Independent Q-learning treats every learning agent as its own learner.

In this project:

- `agent_1` has its own Q-table.
- `agent_2` has its own Q-table.
- Each agent chooses one integer action.
- `MultiAgentRLEnv.step(actions)` applies all actions simultaneously.
- Each Q-table updates from that agent's transition.

There is no shared neural network, no parameter sharing, no centralized critic, and no communication protocol.

## Why MARL Is Harder Than Single-Agent RL

In single-agent RL, the environment is mostly stationary: the agent changes its own policy, but the world rules do not learn.

In MARL, other agents are also learning. That makes the environment non-stationary from each agent's point of view. A state-action pair that looked good early in training may behave differently later because another agent changed its behavior.

This makes learning noisier and coordination harder.

## Decentralized Learning

Phase 5.2 keeps learning decentralized.

The training loop:

- Resets the multi-agent wrapper each episode.
- Gets one observation per controlled agent.
- Asks each `QLearningAgent` for one action.
- Steps all actions together.
- Updates each Q-table independently.
- Tracks shared episode metrics.

The loop returns `MARLTrainingMetrics`.

## Metrics

The current metrics are intentionally simple:

- `success_rate`: fraction of episodes where all controlled agents reached goals.
- `timeout_frequency`: fraction of episodes that timed out.
- `average_episode_length`: average steps per episode.
- `average_reward`: average combined reward per episode.
- `blocked_move_count`: total blocked moves across controlled agents.
- `per_agent_rewards`: per-episode reward totals for each controlled agent.

## Current Limitations

Phase 5.2 does not add:

- Shared rewards.
- Cooperative reward shaping.
- Centralized training.
- Centralized critics.
- Parameter sharing.
- Replay buffers.
- Target networks.
- Curriculum learning.
- Communication learning.
- PPO or DQN.
- PyTorch or TensorFlow.
- RLlib or PettingZoo.

The goal is only to prove that multiple independent tabular agents can train together through the shared environment wrapper.
