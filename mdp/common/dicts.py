from __future__ import annotations

from mdp.common.enums import AlgorithmType, PolicyType

algorithm_name: dict[AlgorithmType, str] = {
    AlgorithmType.DP_POLICY_EVALUATION_V: 'Policy Evaluation DP (V)',
    AlgorithmType.DP_POLICY_IMPROVEMENT_V: 'Policy Improvement DP (V)',
    AlgorithmType.DP_POLICY_ITERATION_V: 'Policy Iteration DP (V)',
    AlgorithmType.DP_VALUE_ITERATION_V: 'Value Iteration DP (V)',

    AlgorithmType.DP_POLICY_EVALUATION_Q: 'Policy Evaluation DP (Q)',
    AlgorithmType.DP_POLICY_IMPROVEMENT_Q: 'Policy Improvement DP (Q)',
    AlgorithmType.DP_POLICY_ITERATION_Q: 'Policy Iteration DP (Q)',

    AlgorithmType.MC_PREDICTION_V: 'MC Prediction (V)',
    AlgorithmType.MC_PREDICTION_Q: 'MC Prediction (Q)',
    AlgorithmType.MC_CONTROL_ON_POLICY: 'On-policy MC Control',

    AlgorithmType.MC_CONSTANT_ALPHA: 'Constant-α MC',
    AlgorithmType.MC_CONTROL_OFF_POLICY: 'Off-policy MC Control',

    AlgorithmType.TD_0: 'TD(0)',
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
