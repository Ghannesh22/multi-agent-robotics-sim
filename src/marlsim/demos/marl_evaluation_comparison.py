from __future__ import annotations

from marlsim.agents import (
    AgentPolicy,
    ConflictAwareWaitingAgent,
    GreedyGoalAgent,
    PriorityBasedCoordinationAgent,
    RandomAgent,
    ShortestPathAgent,
)
from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    MARLRewardMode,
    MultiAgentRLEnv,
    QLearningAgent,
    evaluate_marl_policy,
    format_marl_evaluation_table,
    train_independent_q_learning,
)


def _make_env() -> MultiAgentRLEnv:
    env = GridWorldEnv(
        GridWorldConfig(
            width=3,
            height=2,
            agents=(
                AgentState("agent_1", Position(0, 0), Position(2, 0)),
                AgentState("agent_2", Position(0, 1), Position(2, 1)),
            ),
            max_steps=5,
        )
    )
    return MultiAgentRLEnv(env, ("agent_1", "agent_2"))


def _make_q_agents(env: MultiAgentRLEnv, seed_offset: int = 0) -> dict[str, QLearningAgent]:
    return {
        agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=seed_offset + index)
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }


def _train_q_agents(reward_mode: MARLRewardMode) -> dict[str, QLearningAgent]:
    env = _make_env()
    agents = _make_q_agents(env, seed_offset=10 if reward_mode == MARLRewardMode.SHARED else 0)
    train_independent_q_learning(
        env=env,
        agents=agents,
        episodes=75,
        reward_mode=reward_mode,
    )
    return agents


def _same_policy(env: MultiAgentRLEnv, policy: AgentPolicy) -> dict[str, AgentPolicy]:
    return {agent_id: policy for agent_id in env.controlled_agent_ids}


def run_comparison() -> str:
    individual_agents = _train_q_agents(MARLRewardMode.INDIVIDUAL)
    shared_agents = _train_q_agents(MARLRewardMode.SHARED)

    results = [
        evaluate_marl_policy(
            env=_make_env(),
            policies={
                "agent_1": RandomAgent(seed=1),
                "agent_2": RandomAgent(seed=2),
            },
            episodes=20,
            policy_name="random",
        ),
        evaluate_marl_policy(
            env=_make_env(),
            policies=_same_policy(_make_env(), GreedyGoalAgent()),
            episodes=20,
            policy_name="greedy",
        ),
        evaluate_marl_policy(
            env=_make_env(),
            policies=_same_policy(_make_env(), ShortestPathAgent()),
            episodes=20,
            policy_name="shortest-path",
        ),
        evaluate_marl_policy(
            env=_make_env(),
            policies=_same_policy(_make_env(), ConflictAwareWaitingAgent()),
            episodes=20,
            policy_name="conflict-aware",
        ),
        evaluate_marl_policy(
            env=_make_env(),
            policies=_same_policy(_make_env(), PriorityBasedCoordinationAgent()),
            episodes=20,
            policy_name="priority",
        ),
        evaluate_marl_policy(
            env=_make_env(),
            policies=individual_agents,
            episodes=20,
            policy_name="q-learning individual",
        ),
        evaluate_marl_policy(
            env=_make_env(),
            policies=shared_agents,
            episodes=20,
            policy_name="q-learning shared",
        ),
    ]
    return format_marl_evaluation_table(results)


def main() -> None:
    print("MARL Evaluation Comparison")
    print("==========================")
    print("Policies are evaluated without exploration or learning updates.")
    print()
    print(run_comparison())


if __name__ == "__main__":
    main()
