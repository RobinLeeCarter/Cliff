from __future__ import annotations

from common import enums

algorithm_name: dict[enums.AlgorithmType, str] = {
    enums.AlgorithmType.EXPECTED_SARSA: 'Expected Sarsa',
    enums.AlgorithmType.VQ: 'VQ',
    enums.AlgorithmType.Q_LEARNING: 'Q-learning',
    enums.AlgorithmType.SARSA: 'Sarsa',
    enums.AlgorithmType.CONSTANT_ALPHA_MC: 'Constant-α MC',
    enums.AlgorithmType.TD_0: 'TD(0)'
}

policy_name: dict[enums.PolicyType, str] = {
    enums.PolicyType.DETERMINISTIC: 'Deterministic',
    enums.PolicyType.E_GREEDY: 'ε-greedy',
    enums.PolicyType.NONE: 'No policy',
    enums.PolicyType.RANDOM: 'Random'
}
