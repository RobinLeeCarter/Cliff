from __future__ import annotations

from mdp.common import enums

algorithm_name: dict[enums.AlgorithmType, str] = {
    enums.AlgorithmType.POLICY_EVALUATION_DP_V: 'Policy Evaluation DP (V)',
    enums.AlgorithmType.POLICY_IMPROVEMENT_DP_V: 'Policy Improvement DP (V)',
    enums.AlgorithmType.POLICY_ITERATION_DP_V: 'Policy Iteration DP (V)',
    enums.AlgorithmType.VALUE_ITERATION_DP_V: 'Value Iteration DP (V)',

    enums.AlgorithmType.POLICY_EVALUATION_DP_Q: 'Policy Evaluation DP (Q)',
    enums.AlgorithmType.POLICY_IMPROVEMENT_DP_Q: 'Policy Improvement DP (Q)',
    enums.AlgorithmType.POLICY_ITERATION_DP_Q: 'Policy Iteration DP (Q)',

    enums.AlgorithmType.MC_PREDICTION_V: 'MC Prediction (V)',
    enums.AlgorithmType.MC_PREDICTION_Q: 'MC Prediction (Q)',
    enums.AlgorithmType.ON_POLICY_MC_CONTROL: 'On-policy MC Control',

    enums.AlgorithmType.CONSTANT_ALPHA_MC: 'Constant-α MC',
    enums.AlgorithmType.TD_0: 'TD(0)',
    enums.AlgorithmType.OFF_POLICY_MC_CONTROL: 'Off-policy MC Control',

    enums.AlgorithmType.EXPECTED_SARSA: 'Expected Sarsa',
    enums.AlgorithmType.Q_LEARNING: 'Q-learning',
    enums.AlgorithmType.SARSA: 'Sarsa',
    enums.AlgorithmType.VQ: 'VQ',
}

policy_name: dict[enums.PolicyType, str] = {
    enums.PolicyType.DETERMINISTIC: 'Deterministic',
    enums.PolicyType.E_GREEDY: 'ε-greedy',
    enums.PolicyType.NONE: 'No policy',
    enums.PolicyType.RANDOM: 'Random'
}
