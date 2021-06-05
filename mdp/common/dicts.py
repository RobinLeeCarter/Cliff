from __future__ import annotations

from mdp.common.enums import AlgorithmType, PolicyType, ParallelContextType

algorithm_name: dict[AlgorithmType, str] = {
    AlgorithmType.DP_POLICY_EVALUATION_V_DETERMINISTIC: 'Policy Evaluation DP (V) Deterministic',
    AlgorithmType.DP_POLICY_EVALUATION_V_STOCHASTIC: 'Policy Evaluation DP (V) Stochastic',
    AlgorithmType.DP_POLICY_IMPROVEMENT_V: 'Policy Improvement DP (V)',
    AlgorithmType.DP_POLICY_ITERATION_V: 'Policy Iteration DP (V)',
    AlgorithmType.DP_VALUE_ITERATION_V: 'Value Iteration DP (V)',

    AlgorithmType.DP_POLICY_EVALUATION_Q_DETERMINISTIC: 'Policy Evaluation DP (Q) Deterministic',
    AlgorithmType.DP_POLICY_EVALUATION_Q_STOCHASTIC: 'Policy Evaluation DP (Q) Stochastic',
    AlgorithmType.DP_POLICY_IMPROVEMENT_Q: 'Policy Improvement DP (Q)',
    AlgorithmType.DP_POLICY_ITERATION_Q: 'Policy Iteration DP (Q)',
    AlgorithmType.DP_VALUE_ITERATION_Q: 'Value Iteration DP (Q)',

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

parallel_context_str: dict[ParallelContextType, str] = {
    ParallelContextType.FORK_GLOBAL: 'fork',
    ParallelContextType.FORK_PICKLE: 'fork',
    ParallelContextType.SPAWN: 'spawn',
    ParallelContextType.FORK_SERVER: 'forkserver'
}
