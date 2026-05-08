# Phase 6.5: Communication Evaluation And Comparison

Phase 6.5 adds a systematic comparison layer for communication experiments.

It compares:

- No-communication independent MARL.
- Rule-based communication coordination.
- Communication-aware tabular Q-learning.

This phase is evaluation-focused. It does not add new learning algorithms, change rewards, change `GridWorldEnv`, add communication protocol types, or introduce external dependencies.

## Evaluation Methodology

All modes report a common `CommunicationEvaluationMetrics` structure:

- Success rate.
- Timeout frequency.
- Blocked moves.
- Average reward.
- Average episode length.
- Communication usage count.
- Predicted conflict frequency.
- Waiting frequency.

The no-communication baseline reports zero communication counts. Communication-enabled modes count messages generated during rollout or training.

Predicted conflict frequency is derived from the same symbolic communication features used by Phase 6.4:

```text
predicted_conflict_count / communication_usage_count
```

Waiting frequency is:

```text
waiting_count / communication_usage_count
```

This keeps communication-specific measurement separate from environment rewards.

## Compared Modes

### Independent MARL

Independent MARL uses the existing Phase 5 tabular Q-learning loop.

It has:

- No messages.
- No communication features.
- One Q-table per agent.
- Existing reward logic.

### Rule-Based Communication

Rule-based communication uses `CommunicationAwarePriorityAgent`.

Each step:

1. Agents publish deterministic `CommunicationMessage` objects.
2. Agents choose actions after inspecting messages.
3. The environment applies simultaneous actions normally.
4. Messages and conflict counts are recorded for evaluation.

This mode can perform well in tiny deterministic conflict scenarios because the coordination rule is hand-designed for those conflicts.

### Communication-Aware Q-Learning

Communication-aware Q-learning uses the Phase 6.4 state extension:

```text
(agent_x, agent_y, goal_x, goal_y, other_agent_waiting, other_agent_priority, predicted_conflict, intended_same_target)
```

Agents still train independently with tabular Q-learning. Messages affect the state representation but do not directly choose actions.

## Why Communication Can Hurt Learning

Communication is not automatically useful.

Adding symbolic communication features expands the tabular state space. The learner now needs more experience to estimate Q-values for states that include message conditions. In small training budgets, the extra features can make learning slower or noisier.

Communication can also be weakly useful if the message features do not match the actual decision problem. For example, knowing that a conflict is predicted is useful only if the learner has enough experience to learn when waiting is better than moving.

This explains why initial Phase 6.4 results showed communication-aware Q-learning performing worse than independent MARL in the tiny demo.

## Credit Assignment And State-Space Expansion

Shared rewards already made credit assignment harder in Phase 5.

Communication features add another dimension: an agent must learn whether its own action, another agent's action, or the communicated context caused the outcome. With tabular Q-learning, that can fragment experience across more states.

This phase measures that tradeoff directly instead of assuming communication is beneficial.

## Limits Of Symbolic Communication

The current communication system is deliberately simple:

- Messages are deterministic.
- Features are hand-designed.
- Only current-step communication is encoded.
- There is no learned message policy.
- There is no message history or belief state.
- There is no centralized critic.

This makes the system inspectable, but it also limits expressiveness.

## Demo

Run:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.communication_evaluation_demo
```

The demo prints one comparison table containing:

- No communication independent MARL.
- Rule-based communication coordination.
- Communication-aware Q-learning.

## Future Directions

Later work could explore:

- Better communication feature selection.
- Communication-specific evaluation scenarios.
- Larger training budgets.
- Hybrid rule-based plus learned communication.
- Learned communication policies.
- Centralized critics.
- Graph-based observations.
- Deep MARL.

Those remain out of scope for Phase 6.5.

Phase 6.5 intentionally does not add:

- Deep RL.
- Learned language.
- Transformers.
- Graph neural networks.
- Centralized critics.
- Parameter sharing.
- PyTorch or TensorFlow.
- RLlib or PettingZoo.
- ROS or physics simulation.
