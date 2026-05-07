from __future__ import annotations

from marlsim.core.environment import GridWorldConfig, GridWorldEnv
from marlsim.core.state import AgentState, Position
from marlsim.rl import (
    MARLRewardMode,
    MultiAgentRLEnv,
    QLearningAgent,
    evaluate_marl_policy,
    format_marl_evaluation_table,
    format_marl_trajectory,
    rollout_marl_policy,
    summarize_marl_outcome,
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


def _make_agents(env: MultiAgentRLEnv, seed_offset: int = 0) -> dict[str, QLearningAgent]:
    return {
        agent_id: QLearningAgent(action_size=env.action_size, epsilon=0.2, seed=seed_offset + index)
        for index, agent_id in enumerate(env.controlled_agent_ids, start=1)
    }


def _train_agents(reward_mode: MARLRewardMode) -> dict[str, QLearningAgent]:
    env = _make_env()
    agents = _make_agents(env, seed_offset=10 if reward_mode == MARLRewardMode.SHARED else 0)
    train_independent_q_learning(
        env=env,
        agents=agents,
        episodes=75,
        reward_mode=reward_mode,
    )
    return agents


def build_demo_output() -> str:
    individual_agents = _train_agents(MARLRewardMode.INDIVIDUAL)
    shared_agents = _train_agents(MARLRewardMode.SHARED)

    individual_metrics = evaluate_marl_policy(
        env=_make_env(),
        policies=individual_agents,
        episodes=20,
        policy_name="individual reward",
    )
    shared_metrics = evaluate_marl_policy(
        env=_make_env(),
        policies=shared_agents,
        episodes=20,
        policy_name="shared reward",
    )

    individual_trajectory = rollout_marl_policy(env=_make_env(), policies=individual_agents)
    shared_trajectory = rollout_marl_policy(env=_make_env(), policies=shared_agents)

    return "\n".join(
        (
            "MARL Learning Behavior Demo",
            "===========================",
            "The current individual-reward setup succeeds; the shared-reward setup fails in this toy scenario.",
            "",
            format_marl_evaluation_table([individual_metrics, shared_metrics]),
            "",
            summarize_marl_outcome("individual reward rollout", individual_trajectory),
            format_marl_trajectory(individual_trajectory),
            "",
            summarize_marl_outcome("shared reward rollout", shared_trajectory),
            format_marl_trajectory(shared_trajectory),
            "",
            "Experiment note: the shared reward result is useful because it exposes a credit-assignment problem.",
        )
    )


def main() -> None:
    print(build_demo_output())


if __name__ == "__main__":
    main()
