# Phase 5.5 MARL Visualization and Demos

Phase 5.5 adds readable text demos for multi-agent reinforcement learning behavior.

The goal is not graphical polish. The goal is to make learned behavior inspectable before adding larger MARL experiments.

## What The Demo Shows

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_visualization_demo
```

The demo:

- Trains two independent Q-learning agents with individual rewards.
- Trains two independent Q-learning agents with shared rewards.
- Evaluates both learned policies without exploration.
- Prints a compact comparison table.
- Prints one rollout trajectory for each learned policy.

## Trajectory Format

Each trajectory line shows:

- Step number.
- Agent positions.
- Chosen actions.
- Per-agent rewards.
- Whether each agent was blocked.
- Episode done status.

Example shape:

```text
step 1: agent_1: pos=(1,0) action=right reward=-0.10 blocked=False | agent_2: pos=(1,1) action=right reward=-0.10 blocked=False | done=False
```

This is intentionally plain text so it remains dependency-free and easy to test.

## Current Experiment Result

In the current small two-agent setup:

- Individual reward learning succeeds.
- Shared reward learning fails.

That result is useful. It shows that a team-level reward is not automatically better. Even in a tiny grid, shared rewards can make credit assignment harder because both agents receive the same signal even when one agent's action caused most of the outcome.

## What This Teaches About MARL Reward Design

Independent rewards are easier for each agent to interpret because the reward is tied to that agent's own blocked moves, progress, and goal completion.

Shared rewards can encourage cooperation, but they also blur responsibility. If the team fails, each agent receives the same penalty, even if one agent acted well and another agent got stuck. That can slow or prevent useful learning.

This does not mean shared rewards should be abandoned. It means Phase 5 should treat shared rewards as an experiment that needs careful design, metrics, and scenario coverage.

## Current Limitations

Phase 5.5 does not add:

- Deep RL.
- Neural networks.
- Communication learning.
- Centralized critics.
- Parameter sharing.
- PyTorch or TensorFlow.
- Stable-Baselines3.
- RLlib or PettingZoo.
- Physics simulation.
- ROS.

The demo is a lightweight diagnostic tool for the existing tabular MARL system.
