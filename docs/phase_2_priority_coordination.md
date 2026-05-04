# Phase 2.2: Priority-Based Coordination

Phase 2.2 adds `PriorityBasedCoordinationAgent`, a small rule-based policy that fixes the Phase 2.1 deadlock case.

The policy wraps an existing movement policy, defaulting to `ShortestPathAgent`. It asks the base policy what each agent wants to do next, then resolves obvious conflicts with a stable priority order. By default, agent ids are sorted alphabetically, so `agent_1` has priority over `agent_2`.

Rules:

- If two agents want the same next cell, the higher-priority agent keeps its move.
- The lower-priority agent chooses `STAY`.
- If two agents would directly swap positions, the lower-priority agent yields.
- The higher-priority agent only proceeds when the target is safe. With unchanged Phase 1 environment rules, moving into a yielding agent's current cell is not safe, so the higher-priority agent also waits in that direct-swap case.

This strategy keeps coordination deterministic and readable. It does not change environment rules, add learning, add negotiation, or introduce new planning algorithms. Phase 2.3 can build on this by adding local replanning when waiting is not enough.
