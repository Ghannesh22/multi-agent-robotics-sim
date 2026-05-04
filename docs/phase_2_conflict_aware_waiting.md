# Phase 2.1: Conflict-Aware Waiting

Phase 2.1 adds a simple rule-based coordination policy: `ConflictAwareWaitingAgent`.

The policy wraps an existing movement policy, currently defaulting to `ShortestPathAgent`. It first asks the base policy where the current agent wants to move. Then it estimates each other agent's likely next position using the same base policy. If the current agent's intended move would enter the same next cell as another agent, or create a direct swap, it chooses `STAY` instead.

This keeps the strategy beginner-friendly:

- No environment rules change.
- No reinforcement learning is added.
- No negotiation protocol is added.
- Existing actions, state objects, BFS planning, and environment stepping are reused.

The main tradeoff is that two agents may both wait when they detect the same conflict. That is acceptable for Phase 2.1 because the goal is to avoid obvious blocked moves. Phase 2.2 should handle this limitation with priority-based coordination.
