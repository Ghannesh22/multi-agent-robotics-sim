# Phase 6.3: Rule-Based Communication Experiments

Phase 6.3 adds deterministic communication-aware coordination policies.

This milestone uses the structured `CommunicationMessage` type from Phase 6.1 and the optional communication logging wrapper from Phase 6.2. It does not change reinforcement learning, rewards, training loops, or `GridWorldEnv` movement rules.

## How Policies Consume Messages

The new policies use a two-phase coordination flow:

1. Each agent builds a `CommunicationMessage`.
2. Each agent chooses an action after reading the messages from the other agents.

The message includes the sender, intended action, declared target, waiting status, priority, and step count.

The policies expose:

- `build_message(...)`
- `choose_action_with_messages(...)`
- `predict_conflict_from_messages(...)`

They also still implement `choose_action(...)` so they remain compatible with the existing `AgentPolicy` interface when no messages are supplied.

## CommunicationAwareWaitingAgent

`CommunicationAwareWaitingAgent` uses a base policy, currently `ShortestPathAgent` by default, to compute an intended move.

It waits when received messages predict:

- Same next-cell conflict.
- Direct swap conflict.
- Movement into an agent that has explicitly signaled it is waiting.

If another agent signals waiting and the current agent is not trying to move into that agent's occupied cell, the current agent proceeds with its intended move.

## CommunicationAwarePriorityAgent

`CommunicationAwarePriorityAgent` extends the waiting policy with deterministic priority.

Priority is represented as a non-negative integer in the message. Lower values mean higher priority. By default, sorted agent id order is used. A custom `priority_order` can override that.

The priority policy:

- Lets the higher-priority agent proceed in same-next-cell conflicts.
- Makes the lower-priority agent yield.
- Makes agents wait on direct swap conflicts.
- Uses declared target positions to yield when two agents announce the same target and one has higher priority.

This keeps the logic simple and inspectable.

## Communication Vs No-Communication Coordination

The communication demo compares a no-communication shortest-path step against a communication-aware priority step.

In the toy same-target setup:

- No communication: both agents move toward the same cell and both are blocked by the environment.
- Communication-aware priority: both agents announce intended movement, the lower-priority agent waits, and the higher-priority agent moves without a blocked move.

The environment rules are unchanged in both cases. Communication only changes the rule-based action selected before the environment step.

Run the demo:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.communication_demo
```

## Deterministic Communication Logic

The policies are deterministic:

- Messages are built from the current environment snapshot.
- Intended actions come from the base policy.
- Priority is derived from sorted agent ids or explicit `priority_order`.
- Conflicts are predicted from symbolic intended actions and current positions.

There is no learned communication, random negotiation, language generation, or hidden centralized planner.

## Limitations

Phase 6.3 intentionally does not add:

- Learned communication.
- Neural communication.
- Language generation.
- Communication rewards.
- Emergent language.
- Centralized training.
- Q-learning changes.
- MARL training changes.
- Reward changes.
- `GridWorldEnv` rule changes.
- Networking or distributed simulation.
- PyTorch, TensorFlow, RLlib, PettingZoo, ROS, or physics simulation.

These rule-based policies are a small bridge between message infrastructure and future communication-aware learning experiments.
