# Phase 5.3 Cooperative Reward Experiments

Phase 5.3 compares individual rewards with shared team rewards for independent multi-agent Q-learning.

The environment rules are unchanged. Reward modes are applied inside the MARL training loop before Q-table updates.

## Individual Rewards

Individual reward mode uses each agent's own reward from `MultiAgentRLEnv`.

This includes:

- Step penalty.
- The agent's own blocked-move penalty.
- The agent's own goal reward.
- The agent's own timeout penalty when applicable.

Individual rewards make credit assignment easier because each agent mainly learns from its own outcomes.

Risk: agents may learn selfish behavior and may not naturally coordinate.

## Shared Team Rewards

Shared reward mode gives every controlled agent the same team reward.

The current team reward is intentionally simple:

```text
team_reward = average(per_agent_rewards)
```

If all controlled agents have reached their goals, a small team-completion bonus is added:

```text
team_reward += 5.0
```

Every agent receives that same value for the update.

Shared rewards can help cooperation because all learners receive the same signal for group outcomes.

Risk: shared rewards make credit assignment harder. One agent may receive a good or bad reward mostly caused by another agent's action.

## Metrics

Reward comparison uses the same simple MARL metrics:

- Success rate.
- Timeout frequency.
- Blocked moves.
- Average reward.
- Average episode length.

## Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.marl_reward_comparison
```

The demo trains independent Q-learning agents with individual rewards, then with shared rewards, and prints a comparison table.

## Current Limitations

Phase 5.3 does not add:

- Centralized critics.
- Communication protocols.
- Parameter sharing.
- Cooperative reward shaping beyond the simple team bonus.
- Curriculum learning.
- Replay buffers.
- Target networks.
- PPO or DQN.
- PyTorch or TensorFlow.
- RLlib or PettingZoo.

This phase only compares simple reward modes for independent tabular learners.
