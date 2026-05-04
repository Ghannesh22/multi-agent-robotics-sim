# Phase 2.4: Coordination Metrics

Phase 2.4 adds a repeatable comparison runner for current rule-based policies across current scenarios.

Run it with:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.coordination_comparison
```

The runner uses the existing `GridWorldEnv` and `ExperimentLogger`. It does not change environment rules or add reinforcement learning.

## Compared Policies

- `baseline`: `ShortestPathAgent`
- `waiting`: `ConflictAwareWaitingAgent`
- `priority`: `PriorityBasedCoordinationAgent`
- `replanning`: `LocalReplanningAgent`

## Compared Scenarios

- `simple`
- `narrow_corridor`
- `crossing_paths`

## Results

```text
scenario        | policy     | result  | steps | blocked | blocked reasons | reached goals    | final positions
--------------- | ---------- | ------- | ----- | ------- | --------------- | ---------------- | ----------------------------
simple          | baseline   | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
simple          | waiting    | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
simple          | priority   | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
simple          | replanning | success | 6     | 0       | none            | agent_1, agent_2 | agent_1=(6,0), agent_2=(6,4)
narrow_corridor | baseline   | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
narrow_corridor | waiting    | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
narrow_corridor | priority   | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
narrow_corridor | replanning | success | 10    | 0       | none            | agent_1, agent_2 | agent_1=(6,2), agent_2=(0,4)
crossing_paths  | baseline   | success | 7     | 1       | cell_conflict=1 | agent_1, agent_2 | agent_1=(4,2), agent_2=(2,4)
crossing_paths  | waiting    | failed  | 20    | 0       | none            | none             | agent_1=(1,2), agent_2=(2,1)
crossing_paths  | priority   | success | 7     | 0       | none            | agent_1, agent_2 | agent_1=(4,2), agent_2=(2,4)
crossing_paths  | replanning | success | 8     | 0       | none            | agent_1, agent_2 | agent_1=(4,2), agent_2=(2,4)
```

## Interpretation

The `simple` and `narrow_corridor` scenarios do not currently require extra coordination, so all policies succeed with the same step counts and no blocked moves.

The `crossing_paths` scenario shows the Phase 2 coordination progression:

- The baseline succeeds, but records one `cell_conflict`.
- Conflict-aware waiting avoids blocked moves, but deadlocks because both agents wait.
- Priority-based coordination succeeds and removes the blocked move.
- Local replanning also succeeds with no blocked moves, but takes one extra step because one agent routes around the conflict.

These metrics are intentionally small and console-readable. Phase 2.5 can use them for final polish and documentation.
