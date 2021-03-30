from __future__ import annotations

from mdp.common.comparison_enum import ComparisonType
from mdp.common.enums import AlgorithmType, PolicyType, ScenarioType

algorithm_name: dict[AlgorithmType, str] = {
    AlgorithmType.POLICY_EVALUATION_DP_V: 'Policy Evaluation DP (V)',
    AlgorithmType.POLICY_IMPROVEMENT_DP_V: 'Policy Improvement DP (V)',
    AlgorithmType.POLICY_ITERATION_DP_V: 'Policy Iteration DP (V)',
    AlgorithmType.VALUE_ITERATION_DP_V: 'Value Iteration DP (V)',

    AlgorithmType.POLICY_EVALUATION_DP_Q: 'Policy Evaluation DP (Q)',
    AlgorithmType.POLICY_IMPROVEMENT_DP_Q: 'Policy Improvement DP (Q)',
    AlgorithmType.POLICY_ITERATION_DP_Q: 'Policy Iteration DP (Q)',

    AlgorithmType.MC_PREDICTION_V: 'MC Prediction (V)',
    AlgorithmType.MC_PREDICTION_Q: 'MC Prediction (Q)',
    AlgorithmType.ON_POLICY_MC_CONTROL: 'On-policy MC Control',

    AlgorithmType.CONSTANT_ALPHA_MC: 'Constant-α MC',
    AlgorithmType.TD_0: 'TD(0)',
    AlgorithmType.OFF_POLICY_MC_CONTROL: 'Off-policy MC Control',

    AlgorithmType.EXPECTED_SARSA: 'Expected Sarsa',
    AlgorithmType.Q_LEARNING: 'Q-learning',
    AlgorithmType.SARSA: 'Sarsa',
    AlgorithmType.VQ: 'VQ',
}

policy_name: dict[PolicyType, str] = {
    PolicyType.DETERMINISTIC: 'Deterministic',
    PolicyType.E_GREEDY: 'ε-greedy',
    PolicyType.NONE: 'No policy',
    PolicyType.RANDOM: 'Random'
}

comparison_to_scenario: dict[ComparisonType, ScenarioType] = {
    ComparisonType.BLACKJACK_EVALUATION_V: ScenarioType.BLACKJACK,
    ComparisonType.BLACKJACK_EVALUATION_Q: ScenarioType.BLACKJACK,
    ComparisonType.BLACKJACK_CONTROL_ES: ScenarioType.BLACKJACK,

    ComparisonType.CLIFF_ALPHA_START: ScenarioType.CLIFF,
    ComparisonType.CLIFF_ALPHA_END: ScenarioType.CLIFF,
    ComparisonType.CLIFF_EPISODE: ScenarioType.CLIFF,

    ComparisonType.GAMBLER_VALUE_ITERATION_V: ScenarioType.GAMBLER,

    ComparisonType.JACKS_POLICY_ITERATION_V: ScenarioType.JACKS,
    ComparisonType.JACKS_VALUE_ITERATION_V: ScenarioType.JACKS,
    ComparisonType.JACKS_POLICY_ITERATION_Q: ScenarioType.JACKS,

    ComparisonType.RACETRACK_EPISODE: ScenarioType.RACETRACK,

    ComparisonType.RANDOM_WALK_EPISODE: ScenarioType.RANDOM_WALK,

    ComparisonType.WINDY_TIMESTEP: ScenarioType.RANDOM_WALK,
    ComparisonType.WINDY_TIMESTEP_RANDOM: ScenarioType.WINDY,
}

