from __future__ import annotations
import enum


class EnvironmentType(enum.IntEnum):
    BLACKJACK = enum.auto()
    CLIFF = enum.auto()
    GAMBLER = enum.auto()
    JACKS = enum.auto()
    RACETRACK = enum.auto()
    RANDOM_WALK = enum.auto()
    WINDY = enum.auto()
    MOUNTAIN_CAR = enum.auto()


class ScenarioType(enum.IntEnum):
    BLACKJACK_EVALUATION_V = enum.auto()
    BLACKJACK_EVALUATION_Q = enum.auto()
    BLACKJACK_CONTROL_ES = enum.auto()

    CLIFF_ALPHA_START = enum.auto()
    CLIFF_ALPHA_END = enum.auto()
    CLIFF_EPISODE = enum.auto()

    GAMBLER_VALUE_ITERATION_V = enum.auto()

    JACKS_POLICY_EVALUATION_V = enum.auto()
    JACKS_POLICY_EVALUATION_Q = enum.auto()
    JACKS_POLICY_IMPROVEMENT_V = enum.auto()
    JACKS_POLICY_IMPROVEMENT_Q = enum.auto()
    JACKS_POLICY_ITERATION_V = enum.auto()
    JACKS_POLICY_ITERATION_Q = enum.auto()
    JACKS_VALUE_ITERATION_V = enum.auto()
    JACKS_VALUE_ITERATION_Q = enum.auto()

    RACETRACK_EPISODE = enum.auto()

    RANDOM_WALK_EPISODE = enum.auto()

    WINDY_TIMESTEP = enum.auto()
    WINDY_TIMESTEP_RANDOM = enum.auto()


class Square:       # (enum.IntEnum) no longer an enum for 3x-30x speed improvement
    NORMAL: int = 0
    CLIFF: int = 1   # or grass
    START: int = 2
    END: int = 3


class BreakdownType(enum.IntEnum):
    EPISODE_BY_TIMESTEP = enum.auto()
    RETURN_BY_EPISODE = enum.auto()
    RMS_BY_EPISODE = enum.auto()
    RETURN_BY_ALPHA = enum.auto()


class AlgorithmType(enum.IntEnum):
    """parent enum for TabularAlgorithmType and NonTabularAlgorithmType"""


class TabularAlgorithmType(AlgorithmType):
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


class PolicyType(enum.IntEnum):
    DETERMINISTIC = enum.auto()
    E_GREEDY = enum.auto()
    NONE = enum.auto()
    RANDOM = enum.auto()


class DualPolicyRelationship(enum.IntEnum):
    SINGLE_POLICY = enum.auto()
    LINKED_POLICIES = enum.auto()
    INDEPENDENT_POLICIES = enum.auto()


class ActionsList(enum.IntEnum):
    NO_ACTIONS = enum.auto()
    FOUR_MOVES = enum.auto()
    FOUR_CLIFF_FRIENDLY_MOVES = enum.auto()
    KINGS_MOVES = enum.auto()
    KINGS_MOVES_PLUS_NO_MOVE = enum.auto()


class GridViewType(enum.IntEnum):
    POSITION = enum.auto()
    POSITION_MOVE = enum.auto()
    JACKS = enum.auto()
    BLACKJACK = enum.auto()


class UserEvent(enum.IntEnum):
    NONE = enum.auto()
    QUIT = enum.auto()
    SPACE = enum.auto()


class ParallelContextType(enum.IntEnum):
    NONE = enum.auto()
    FORK_GLOBAL = enum.auto()
    FORK_PICKLE = enum.auto()
    SPAWN = enum.auto()
    FORK_SERVER = enum.auto()
