# Phase 5 Summary

Phase 5 completed the project's first multi-agent reinforcement learning milestone.

The milestone extended the Phase 4 tabular Q-learning baseline from one learning-controlled agent to multiple learning-controlled agents in the same `GridWorldEnv`. It kept the system lightweight, reproducible, and inspectable. It did not change the core grid-world movement, collision, obstacle, or goal rules.

## What Phase 5 Achieved

- Added the Phase 5 MARL plan in `docs/phase_5_plan.md`.
- Added `MultiAgentRLEnv`.
- Added per-agent observations, actions, rewards, and info dictionaries.
- Added independent multi-agent Q-learning training.
- Added individual reward mode.
- Added shared/team reward mode.
- Added MARL evaluation helpers.
- Added MARL metrics for success, timeout, blocked moves, rewards, and episode length.
- Added text trajectory visualization.
- Added demos for training, reward comparison, evaluation comparison, and learned behavior.
- Added documentation and tests for the multi-agent wrapper, training, rewards, evaluation, and visualization.

## How MultiAgentRLEnv Works

`MultiAgentRLEnv` wraps the existing `GridWorldEnv`.

The wrapper controls multiple selected agent ids:

```python
MultiAgentRLEnv(env, ("agent_1", "agent_2"))
```

`reset()` returns one observation per controlled agent:

```python
{
    "agent_1": (agent_x, agent_y, goal_x, goal_y),
    "agent_2": (agent_x, agent_y, goal_x, goal_y),
}
```

`step(actions)` accepts one integer action per controlled agent:

```python
{
    "agent_1": 3,
    "agent_2": 1,
}
```

The integer actions reuse the Phase 3 action mapping:

| Integer | Action |
| ------- | ------ |
| `0` | up |
| `1` | down |
| `2` | left |
| `3` | right |
| `4` | stay |

The wrapper calls `GridWorldEnv.step(...)` with all actions at once, so movement and collision behavior still come from the original simulator.

The result is:

```python
observations, rewards, done, info
```

Rewards are returned per agent, and `done` is a shared episode flag for the controlled group.

## How Independent Q-Learning Works

Phase 5 uses independent Q-learning.

Each controlled learning agent owns a separate `QLearningAgent` and a separate Q-table:

```text
agent_1 -> q_table_1
agent_2 -> q_table_2
```

For each training step:

- Each agent observes its own coordinate vector.
- Each agent chooses an integer action with epsilon-greedy exploration.
- `MultiAgentRLEnv.step(actions)` applies all actions simultaneously.
- Each Q-table updates independently from that agent's transition.

There is no parameter sharing, centralized critic, neural network, replay buffer, target network, or communication protocol.

## Reward Modes

Phase 5 compares two reward modes.

Individual reward mode uses each agent's own environment reward:

- Small step penalty.
- The agent's own blocked-move penalty.
- The agent's own goal reward.
- The agent's own timeout penalty.

Shared reward mode gives all controlled agents the same team reward:

```text
team_reward = average(per_agent_rewards)
```

When all controlled agents reach their goals, the team receives an additional completion bonus.

## Individual Vs Shared Reward Results

In the current small two-agent scenario, individual rewards work well.

The learned individual-reward policy reaches both goals:

```text
individual reward | success 1.00 | timeout 0.00 | average steps 2.00
```

The shared-reward policy currently fails in the same toy setup:

```text
shared reward | success 0.00 | timeout 1.00 | average steps 5.00
```

This is an experiment result, not a hidden failure.

## Why Shared Reward Failed

The shared reward makes credit assignment harder.

When every agent receives the same signal, each learner has less direct evidence about whether its own action helped or hurt. In the current tiny scenario, one agent can learn behavior that looks locally stable while the team still fails to complete the episode. Because both agents receive the same team-level signal, the useful action attribution is blurred.

This shows that shared rewards are not automatically better for cooperation. They may need better shaping, richer observations, more episodes, different exploration settings, or communication before they become useful.

## What This Teaches About Credit Assignment

MARL is harder than single-agent RL because each learner changes the environment seen by the others.

Credit assignment asks: "Which agent's action caused this outcome?"

Individual rewards make that question easier because each agent receives feedback tied to its own movement, blocked moves, goal completion, and timeout.

Shared rewards can encourage team behavior, but they also make responsibility less clear. Phase 5 demonstrates this tradeoff in a small reproducible setup before moving to more complex coordination mechanisms.

## Evaluation And Visualization

Phase 5 added `evaluate_marl_policy(...)` for fixed-policy evaluation.

During evaluation:

- Q-learning exploration is disabled.
- Original epsilon values are restored afterward.
- Q-tables are not updated.
- Baselines and learned policies use the same metrics.

Phase 5 also added text trajectory output. Each rollout line shows:

- Step number.
- Agent positions.
- Actions.
- Rewards.
- Blocked status.
- Done status.

This keeps learned behavior readable without adding plotting or dashboard dependencies.

## Demos

Run independent MARL training:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_training_demo
```

Run individual vs shared reward comparison:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_reward_comparison
```

Run policy evaluation comparison:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_evaluation_comparison
```

Run learned-behavior trajectory visualization:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_visualization_demo
```

## Why Phase 6 Communication Comes Next

Phase 5 shows that independent learners can train together, but it also exposes a core MARL limitation: agents do not know each other's intent.

Phase 6 should explore simple communication signals, such as sharing position or intended movement. Communication can help agents coordinate before conflicts happen and may make cooperative reward designs easier to learn from.

Phase 6 should still stay grid-world first and should not jump to deep RL or physics simulation.

## Intentionally Out Of Scope

Phase 5 intentionally did not add:

- Deep RL.
- Neural networks.
- PyTorch.
- TensorFlow.
- Stable-Baselines3.
- RLlib.
- PettingZoo.
- PPO.
- DQN.
- Centralized critics.
- Parameter sharing.
- Replay buffers.
- Target networks.
- Communication learning.
- PyBullet.
- ROS.
- Physics simulation.

The milestone stays focused on simple, tabular, independent MARL in the existing grid-world.
