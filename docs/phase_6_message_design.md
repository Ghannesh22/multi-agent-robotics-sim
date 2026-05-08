# Phase 6.1: Communication Message Design

Phase 6.1 defines the structured message format agents can use in later communication experiments.

This milestone adds the message data structure only. It does not change learning behavior, reward logic, MARL training, action selection, or `GridWorldEnv` movement rules.

## Why Explicit Communication Matters

Phase 5 showed that independent multi-agent Q-learning can train multiple agents in the same grid-world, but shared rewards alone did not produce reliable coordination in the toy cooperative experiment.

The main issue is information. A shared reward tells every agent how the team did, but it does not tell an individual agent what the other agents intended to do. If two agents block each other, both may receive weak or delayed feedback without knowing whether the other agent planned to move, wait, yield, or claim priority.

Explicit communication gives agents a small amount of structured coordination context before future action-selection experiments.

## What Agents Share

The first communication message is `CommunicationMessage`.

It contains:

- `sender_id`: the agent that produced the message.
- `intended_action`: the action the sender expects to take.
- `target_position`: the sender's current goal, waypoint, or temporary target.
- `blocked`: whether the sender is blocked or reporting a blocked state.
- `waiting`: whether the sender is intentionally waiting.
- `priority`: a simple non-negative priority signal for tie-breaking.
- `step_count`: the environment step associated with the message.

The message is intentionally symbolic. It shares compact coordination facts, not free-form language.

Messages are currently modeled as broadcast-style coordination facts. They do not include a recipient field yet. Future wrappers can decide whether every controlled agent sees every message or whether messages are filtered before action selection.

The default no-message behavior is represented by missing messages or by default fields such as `intended_action=None`, `target_position=None`, `blocked=False`, `waiting=False`, and `priority=0`.

## Why Communication Is Structured

Structured messages are easier to test, print, serialize, and eventually include in tabular observations.

They also keep the communication contract clear:

- Message fields have explicit names.
- Message values are simple Python values or existing project types.
- Formatting is deterministic.
- Validation can catch malformed messages before they are used by later wrappers.

This avoids hidden protocol behavior and keeps future experiments reproducible.

## Why The Design Is Simple First

The project is still using lightweight grid-world MARL and tabular Q-learning. A simple message format is enough to test whether explicit intent sharing helps before adding larger systems.

The first design avoids:

- Negotiation protocols.
- Learned language.
- Neural message encoders.
- Distributed networking.
- External MARL libraries.

This keeps Phase 6 focused on the research question: does explicit symbolic communication improve coordination compared with no communication?

## Helper Utilities

Phase 6.1 adds:

- `serialize_message(...)`: converts a message into built-in Python values.
- `format_message(...)`: returns a compact readable string for logs and demos.
- `validate_message(...)`: returns a list of validation errors, or an empty list for a valid message.

Example formatted output:

```text
step=5 sender=agent_1 action=right target=(2, 0) blocked=False waiting=False priority=1
```

Example serialized output:

```python
{
    "sender_id": "agent_1",
    "intended_action": "right",
    "target_position": {"x": 2, "y": 0},
    "blocked": False,
    "waiting": False,
    "priority": 1,
    "step_count": 5,
}
```

## Current Limitations

Phase 6.1 does not add:

- Communication-aware Q-learning.
- Communication policies.
- Communication-aware action selection.
- Reward changes.
- MARL training changes.
- Environment rule changes.
- Networking or distributed systems.
- Natural language generation.
- Learned language.
- Transformers.
- LLM communication.

The message structure is a foundation for later subphases. Phase 6.2 can use it to design a communication-enabled MARL wrapper, and Phase 6.3 can use it for rule-based communication experiments.
