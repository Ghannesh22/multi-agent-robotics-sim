# Phase 6.6: Communication Visualization And Trajectory Demos

Phase 6.6 makes communication behavior inspectable through dependency-free text trajectories.

The goal is not to add graphics. The goal is to show what each agent saw, said, and did during communication-based coordination and communication-aware learning.

## Why Visualization Matters For MARL Debugging

Multi-agent reinforcement learning is difficult to debug from aggregate metrics alone.

A table can show that one mode had more blocked moves or lower success, but it does not show why. Communication trajectories make the local behavior visible:

- Which messages were sent.
- Which action each agent selected.
- Whether an agent waited.
- Whether a conflict was predicted.
- Whether the environment blocked a move.
- Whether the episode finished or timed out.

This is especially useful after Phase 6.5, where communication-aware Q-learning underperformed the simpler baselines in the tiny demo.

## Visualization Helpers

Phase 6.6 adds:

- `format_communication_rollout(...)`
- `format_message_timeline(...)`
- `summarize_communication_episode(...)`
- `visualize_waiting_behavior(...)`
- `visualize_conflict_predictions(...)`

The rollout helpers support:

- Independent MARL with no communication.
- Rule-based communication coordination.
- Communication-aware Q-learning.

## How Communication Behavior Is Interpreted

Each step records:

- Agent positions.
- Actions.
- Rewards.
- Sent messages.
- Waiting flags.
- Priority values.
- Predicted conflicts.
- Blocked moves.
- Goal completion.
- Done and timeout state.

Example:

```text
STEP 2 done=False timeout=False
agent_1:
  pos=(1,0)
  action=right
  reward=-0.10
  blocked=False
  reached_goal=False
  sent="step=1 sender=agent_1 action=right target=(2, 0) blocked=False waiting=False priority=0"
  waiting=False
  priority=0
  predicted_conflict=False
```

## Deterministic Vs Learned Coordination Traces

Rule-based communication traces are deterministic and easy to interpret. If a lower-priority agent waits, the reason is visible in the message and priority fields.

Communication-aware Q-learning traces are more subtle. The message features are part of the learned state, but the Q-table still chooses the action. This can expose:

- Useful waiting.
- Unnecessary waiting.
- Missed conflict predictions.
- Learned behavior that underperforms a simple deterministic rule.

That distinction is important: communication data being available does not mean the learner will use it well.

## Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.communication_visualization_demo
```

The demo prints:

- A summary for each mode.
- A detailed rollout for each mode.
- Message timelines for communication-enabled modes.
- Waiting behavior traces.
- Conflict prediction traces.
- A short observation note connecting the trajectory to the Phase 6.5 findings.

## Limitations Of Text-Based Visualization

The visualization is intentionally lightweight.

It does not include:

- Pygame.
- Matplotlib.
- Animations.
- Web dashboards.
- Graphical UI.
- Networking visualization.

Text output is less visually rich than a rendered grid, but it is fast, testable, dependency-free, and precise enough for debugging small MARL communication experiments.

Phase 6.6 intentionally does not change:

- `GridWorldEnv` rules.
- Reward logic.
- Learning algorithms.
- Communication protocol types.
- Training behavior.
