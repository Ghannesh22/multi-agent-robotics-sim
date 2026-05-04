# Phase 2.3: Local Replanning

Phase 2.3 adds `LocalReplanningAgent`, a simple coordination policy that tries an alternate route when its preferred next move is unsafe.

The policy builds on priority-based coordination and reuses the existing BFS shortest-path planner. It first asks the base policy, currently `ShortestPathAgent`, for the preferred move. If that move is safe, the agent uses it. If the preferred next cell is unsafe because of a same-cell conflict, direct-swap conflict, or occupied stationary cell, the policy temporarily treats unsafe cells and current agent positions as blocked and asks BFS for another path to the same goal.

If BFS finds an alternate path, the agent takes the first step on that path. If no safe alternate path exists, the agent chooses `STAY`.

This remains local and deterministic:

- No environment rules change.
- No reinforcement learning is added.
- No global negotiation or reservation table is added.
- Existing actions, state objects, priority logic, and BFS planning are reused.

The goal is not perfect traffic control. The goal is to reduce unnecessary waiting in small bottleneck or crossing cases while keeping the rule easy to read and test.
