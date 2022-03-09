import enum


# TODO: Make TabularAlgorithmType? Might make comparison typing too hard as can't have a common parent class
class AlgorithmType(enum.IntEnum):
    DP_POLICY_EVALUATION_V_DETERMINISTIC = enum.auto()
    DP_POLICY_EVALUATION_V_STOCHASTIC = enum.auto()
    DP_POLICY_IMPROVEMENT_V = enum.auto()
    DP_POLICY_ITERATION_V = enum.auto()
    DP_VALUE_ITERATION_V = enum.auto()

    DP_POLICY_EVALUATION_Q_DETERMINISTIC = enum.auto()
    DP_POLICY_EVALUATION_Q_STOCHASTIC = enum.auto()
    DP_POLICY_IMPROVEMENT_Q = enum.auto()
    DP_POLICY_ITERATION_Q = enum.auto()
    DP_VALUE_ITERATION_Q = enum.auto()

    MC_PREDICTION_V = enum.auto()
    MC_PREDICTION_Q = enum.auto()
    MC_CONTROL_ON_POLICY = enum.auto()

    MC_CONSTANT_ALPHA = enum.auto()
    MC_CONTROL_OFF_POLICY = enum.auto()

    TD_0 = enum.auto()
    SARSA = enum.auto()
    Q_LEARNING = enum.auto()
    EXPECTED_SARSA = enum.auto()
    VQ = enum.auto()
