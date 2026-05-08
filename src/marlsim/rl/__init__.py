"""Reinforcement-learning-style wrappers for the grid-world simulator."""

from marlsim.rl.q_learning import QLearningAgent
from marlsim.rl.communicating_multi_agent_env import CommunicatingMultiAgentRLEnv
from marlsim.rl.communication_q_learning import (
    CommunicationAwareQLearningAgent,
    build_communication_observation,
    extend_observation_with_messages,
    train_communication_q_learning,
)
from marlsim.rl.communication_evaluation import (
    CommunicationEvaluationMetrics,
    compare_communication_modes,
    evaluate_communication_q_learning,
    evaluate_independent_marl,
    evaluate_rule_based_communication,
    format_communication_table,
    summarize_communication_metrics,
)
from marlsim.rl.communication_visualization import (
    CommunicationTrajectoryStep,
    format_communication_rollout,
    format_message_timeline,
    rollout_communication_q_learning_trace,
    rollout_independent_communication_trace,
    rollout_rule_based_communication_trace,
    summarize_communication_episode,
    visualize_conflict_predictions,
    visualize_waiting_behavior,
)
from marlsim.rl.evaluation import EvaluationMetrics, evaluate_policy, format_evaluation_table
from marlsim.rl.marl_training import (
    MARLRewardMode,
    MARLTrainingMetrics,
    apply_reward_mode,
    summarize_marl_training,
    train_independent_q_learning,
)
from marlsim.rl.marl_visualization import (
    MARLTrajectoryStep,
    format_marl_trajectory,
    rollout_marl_policy,
    summarize_marl_outcome,
)
from marlsim.rl.marl_evaluation import (
    MARLEvaluationMetrics,
    evaluate_marl_policy,
    format_marl_evaluation_table,
)
from marlsim.rl.multi_agent_env import MultiAgentRLEnv
from marlsim.rl.single_agent_env import RewardConfig, SingleAgentRLEnv
from marlsim.rl.training import (
    TrainingMetrics,
    export_training_metrics,
    reward_chart,
    success_rate_chart,
    summarize_training,
    train_q_learning,
)

__all__ = [
    "EvaluationMetrics",
    "CommunicationAwareQLearningAgent",
    "CommunicationEvaluationMetrics",
    "CommunicationTrajectoryStep",
    "CommunicatingMultiAgentRLEnv",
    "QLearningAgent",
    "MARLRewardMode",
    "MARLEvaluationMetrics",
    "MARLTrajectoryStep",
    "MARLTrainingMetrics",
    "MultiAgentRLEnv",
    "RewardConfig",
    "SingleAgentRLEnv",
    "TrainingMetrics",
    "apply_reward_mode",
    "build_communication_observation",
    "compare_communication_modes",
    "evaluate_communication_q_learning",
    "evaluate_independent_marl",
    "evaluate_policy",
    "evaluate_marl_policy",
    "evaluate_rule_based_communication",
    "extend_observation_with_messages",
    "export_training_metrics",
    "format_evaluation_table",
    "format_communication_table",
    "format_communication_rollout",
    "format_marl_evaluation_table",
    "format_marl_trajectory",
    "format_message_timeline",
    "reward_chart",
    "rollout_communication_q_learning_trace",
    "rollout_independent_communication_trace",
    "rollout_marl_policy",
    "rollout_rule_based_communication_trace",
    "success_rate_chart",
    "summarize_communication_episode",
    "summarize_marl_training",
    "summarize_marl_outcome",
    "summarize_communication_metrics",
    "summarize_training",
    "train_communication_q_learning",
    "train_independent_q_learning",
    "train_q_learning",
    "visualize_conflict_predictions",
    "visualize_waiting_behavior",
]
