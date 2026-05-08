# Phase 6 Plan: Communication Between Agents

## 1. Phase 6 Vision

Phase 6 explores whether explicit communication between agents can improve coordination in the existing multi-agent reinforcement learning system.

Phase 5 proved that multiple tabular Q-learning agents can train together in the same `GridWorldEnv`. It also exposed a coordination failure: shared rewards alone did not reliably produce cooperative behavior in the small two-agent experiment. The agents learned from team-level reward signals, but they still had no direct way to know what the other agent intended to do.

Communication is the next lightweight coordination mechanism to study before adding more complex learning algorithms or external MARL frameworks.

The goal is not to add natural language, neural communication, or deep MARL. The goal is to design a small, inspectable communication layer that can later be tested with rule-based policies and tabular Q-learning.

## 2. Why Communication Matters In MARL

Multi-agent reinforcement learning is difficult because each agent learns while the other agents are also acting and learning. From one agent's perspective, the environment is not fully stationary. Other agents can block paths, change which moves are useful, create conflicts, or solve part of the task before the local agent receives useful feedback.

Without communication, each learner only sees its own observation and reward signal. In the current project, that means each learning agent observes its own position and goal, chooses an action, and receives reward after the joint environment step. The agent can infer some coordination pressure from blocked moves or timeouts, but it cannot directly observe intent.

Simple communication can make hidden coordination information explicit:

- Which move an agent plans to take next.
- Which target or goal an agent is pursuing.
- Whether an agent is blocked.
- Whether an agent is waiting intentionally.
- Which agent should have priority in a conflict.

This can help agents avoid conflicts before they happen instead of only learning from blocked moves after the fact.

## 3. Why Shared Rewards Alone Were Insufficient

Phase 5 compared individual rewards and shared rewards.

Individual rewards worked well in the current toy scenario because each agent received feedback tied to its own movement, blocked moves, goal completion, and timeout outcome.

Shared rewards failed in the same setup because the team-level reward blurred credit assignment. When all agents receive the same reward, each learner has weaker evidence about whether its own action helped or hurt the team. A shared reward can say that the team outcome was bad, but it does not say which agent should have moved, waited, yielded, or changed path.

This matters because cooperation requires more than aligned incentives. The agents also need enough information to select compatible actions. Shared reward aligns the objective, but it does not expose the other agent's plan.

Phase 6 treats communication as a way to add coordination information while keeping the learning algorithm simple and tabular.

## 4. How Communication May Improve Cooperation

Communication may improve cooperation by reducing uncertainty before action selection.

Expected benefits:

- Fewer same-cell and direct-swap conflicts.
- Fewer repeated blocked moves.
- Less deadlock in narrow corridors or crossing paths.
- Better waiting behavior when one agent should yield.
- More stable cooperative behavior under shared or hybrid rewards.
- Easier interpretation of why a policy chose to move or wait.

Communication should be evaluated as an experimental variable:

```text
no communication policy
vs
communication-aware policy
```

The comparison should use the same scenarios, seeds, metrics, and reward modes where possible.

## 5. Decentralized Coordination Concepts

Phase 6 should preserve decentralized decision-making.

Each agent should still choose its own action. Communication should add a small amount of shared information, not a central controller that directly commands all agents.

Useful decentralized coordination concepts:

- Local policy: each agent maps its own observation plus received messages to an action.
- Peer messages: agents publish simple structured signals for other agents to observe.
- Simultaneous action: the environment still applies all actions together through `GridWorldEnv.step(...)`.
- Local responsibility: each agent remains responsible for its own movement decision.
- Coordination without central control: communication helps agents align behavior without replacing their policies.

This keeps Phase 6 aligned with the current independent MARL architecture.

## 6. Intent Sharing Concepts

Intent sharing means exposing what an agent is trying to do before the joint step is executed.

In the current environment, conflicts happen when agents select incompatible movements at the same time. If an agent can announce its intended next move or current target, another agent can decide to wait, take a different route, or claim priority.

Planned intent signals should be simple and symbolic. They should be inspectable in logs and easy to include in tabular observations later.

Examples:

- Agent 1 intends to move right.
- Agent 2 is targeting its goal cell.
- Agent 1 is blocked and may need another agent to move first.
- Agent 2 is waiting this turn.
- Agent 1 has priority in a narrow corridor.

The first communication experiments should prefer fixed, rule-based message construction before adding communication-aware learning.

## 7. Explicit Vs Implicit Coordination

Implicit coordination happens when agents adapt to each other without a direct message channel. Phase 2 rule-based policies and Phase 5 independent MARL mostly rely on implicit coordination. Agents react to positions, obstacles, blocked moves, or learned reward patterns.

Explicit coordination happens when agents exchange structured information before choosing or executing actions. A message such as "I intend to move right" or "I am waiting" gives another agent information that is not available from position alone.

Phase 6 should compare both:

- Implicit coordination baseline: existing rule-based or learned behavior with no messages.
- Explicit coordination variant: the same scenario with structured messages available.

The purpose is to measure whether messages improve behavior, not to assume communication is automatically better.

## 8. Planned Message Types

Phase 6 should start with a small message vocabulary.

### Intended Next Move

Represents the action an agent expects to take on the next step.

Example values:

- `up`
- `down`
- `left`
- `right`
- `stay`

Use cases:

- Detect likely same-cell conflicts.
- Detect likely direct swaps.
- Allow another agent to yield before a blocked move occurs.

### Current Target

Represents the cell or goal an agent is currently pursuing.

Example values:

- Own goal position.
- Intermediate waypoint.
- Temporary detour target.

Use cases:

- Make goal-directed behavior easier to interpret.
- Support future path or corridor coordination.
- Distinguish purposeful waiting from target confusion.

### Blocked Status

Represents whether the agent was blocked recently or expects to be blocked.

Example values:

- `blocked`
- `not_blocked`

Use cases:

- Detect deadlock-like patterns.
- Let nearby agents yield when another agent is stuck.
- Support metrics for communication-triggered conflict resolution.

### Wait Signal

Represents that an agent intends to wait intentionally.

Example values:

- `waiting`
- `not_waiting`

Use cases:

- Avoid interpreting all `stay` actions as failure.
- Allow one agent to give another agent space.
- Make learned waiting behavior easier to visualize.

### Priority Signal

Represents which agent should move first when two actions conflict.

Example values:

- `has_priority`
- `yielding`
- `neutral`

Use cases:

- Coordinate narrow corridors.
- Break symmetry in crossing paths.
- Compare communication-aware priority rules against Phase 2 priority coordination.

## 9. Phase 6 Subphases

### Phase 6.0: Communication Planning

Status: this document.

- Define why communication is being explored.
- Document the shared reward and credit assignment motivation.
- Define the initial message vocabulary.
- Define planned architecture boundaries.
- Define what remains out of scope.
- No implementation.

### Phase 6.1: Communication Message Design

Status: completed in `docs/phase_6_message_design.md`.

- Define a structured message representation.
- Decide whether messages are dictionaries, dataclasses, tuples, or enums.
- Document sender, recipient scope, message fields, and default no-message behavior.
- Decide how messages should be logged for debugging.
- Keep message design independent from learning updates.
- Do not add new dependencies.

### Phase 6.2: Communication-Enabled MARL Wrapper

Status: completed in `docs/phase_6_communication_wrapper.md`.

- Plan and implement a wrapper layer that exposes messages to agents.
- Preserve `GridWorldEnv` as the source of movement and collision truth.
- Preserve simultaneous environment stepping.
- Avoid changing reward logic in this subphase.
- Keep communication optional so no-communication baselines still work.

### Phase 6.3: Rule-Based Communication Experiments

Status: completed in `docs/phase_6_rule_based_communication.md`.

- Build simple rule-based message producers and consumers.
- Test intended-move sharing, wait signals, and priority signals before learning from messages.
- Compare communication rules against existing Phase 2 coordination baselines.
- Use small deterministic scenarios first.

### Phase 6.4: Communication-Aware Q-Learning Agents

Status: completed in `docs/phase_6_communication_q_learning.md`.

- Extend tabular observations to include simple communication fields.
- Keep one Q-table per controlled agent.
- Keep independent learning.
- Avoid neural networks and centralized critics.
- Compare communication-aware Q-learning against Phase 5 independent Q-learning.

### Phase 6.5: Communication Evaluation And Comparison

Status: completed in `docs/phase_6_communication_evaluation.md`.

- Compare no-communication and communication-enabled runs.
- Track success rate, timeout rate, blocked moves, average steps, and total reward.
- Track communication-specific metrics such as wait-signal frequency and conflict avoidance.
- Reuse existing evaluation patterns where possible.

### Phase 6.6: Visualization Of Communication Behavior

Status: completed in `docs/phase_6_communication_visualization.md`.

- Add text output that shows messages alongside positions, actions, rewards, and blocked status.
- Make communication behavior inspectable without adding plotting dependencies.
- Show whether agents waited, yielded, claimed priority, or changed action after receiving messages.

### Phase 6.7: Final Phase 6 Polish And Stabilization

Status: completed in `docs/phase_6_summary.md`.

- Update README and project documentation.
- Run all tests and demos relevant to Phase 6.
- Document experiment results and remaining limitations.
- Prepare final Phase 6 completion summary and tag recommendation.

## 10. Architecture Boundaries

Phase 6 should keep the existing architecture stable.

Rules:

- Do not modify core grid-world movement rules for communication.
- Do not make communication a hidden side effect inside `GridWorldEnv`.
- Do not replace simultaneous stepping.
- Do not add external MARL frameworks.
- Do not require deep learning libraries.
- Keep no-communication behavior available as a baseline.
- Keep messages small, structured, and printable.

Communication should be modeled as information available to agents or wrappers, not as a change to the physics or collision rules of the environment.

## 11. Initial Evaluation Questions

Phase 6 should answer:

- Does explicit communication reduce blocked moves?
- Does it reduce timeouts in narrow corridor or crossing-path scenarios?
- Does it help shared reward training avoid the Phase 5 failure mode?
- Do rule-based communication policies outperform no-communication baselines?
- Does adding messages to tabular observations make Q-learning better or just increase state size?
- Which message types provide useful coordination signal?
- Which message types add complexity without improving outcomes?

## 12. Risks

### State Space Growth

Adding messages to observations can increase the number of tabular states. This can make Q-learning slower or less stable.

Mitigation:

- Start with small symbolic message values.
- Add one message type at a time.
- Track Q-table size and state coverage.

### Over-Engineering The Protocol

A rich communication protocol could distract from the core research question.

Mitigation:

- Start with intended next move, blocked status, wait signal, and priority signal.
- Keep messages easy to print and test.
- Avoid negotiation systems in the first implementation.

### Hidden Centralization

Communication experiments can accidentally become centralized planning if one component decides all actions.

Mitigation:

- Keep each agent responsible for its own action.
- Use messages as inputs, not commands.
- Preserve decentralized action selection.

### Confusing Communication With Reward Changes

Changing rewards at the same time as communication would make results hard to interpret.

Mitigation:

- Hold reward logic fixed when first testing communication.
- Compare against Phase 5 reward modes explicitly.
- Change one experimental variable at a time.

## 13. Intentionally Out Of Scope

Phase 6 should not include:

- Natural language communication.
- LLM agents.
- Transformers.
- Deep MARL.
- Centralized critics.
- Graph neural networks.
- PyTorch.
- TensorFlow.
- RLlib.
- PettingZoo.
- ROS.
- Physics simulation.

These may be interesting future directions, but they are not needed for a lightweight, inspectable communication milestone.

## 14. Success Criteria

Phase 6 should be considered successful when:

- Communication message types are clearly designed and documented.
- Communication can be enabled or disabled for comparison.
- Rule-based communication experiments are reproducible.
- Communication-aware Q-learning remains tabular and inspectable.
- Evaluation compares communication against no-communication baselines.
- Visualization makes communication behavior easy to understand.
- Existing Phase 5 behavior remains available and tested.

## 15. Next Implementation Step

Phase 6 is complete.

The next milestone is Phase 7: Physics-Based Simulation Upgrade. Phase 7 should not start until the Phase 6 changes are committed and tagged.
