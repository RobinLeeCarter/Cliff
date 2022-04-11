from __future__ import annotations
import enum


class EnvironmentType(enum.IntEnum):
    BLACKJACK = enum.auto()
    CLIFF = enum.auto()
    GAMBLER = enum.auto()
    JACKS = enum.auto()
    MOUNTAIN_CAR = enum.auto()
    RACETRACK = enum.auto()
    RANDOM_WALK = enum.auto()
    WINDY = enum.auto()


class ComparisonType(enum.IntEnum):
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

    MOUNTAIN_CAR_STANDARD = enum.auto()
    MOUNTAIN_CAR_SERIAL_BATCH = enum.auto()
    MOUNTAIN_CAR_PARALLEL_BATCH = enum.auto()

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
    # Tabular
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

    TABULAR_TD_0 = enum.auto()
    TABULAR_SARSA = enum.auto()
    TABULAR_Q_LEARNING = enum.auto()
    TABULAR_EXPECTED_SARSA = enum.auto()
    TABULAR_VQ = enum.auto()

    # Non-tabular
    NON_TABULAR_EPISODIC_SARSA = enum.auto()
    NON_TABULAR_EPISODIC_SARSA_BATCH = enum.auto()
    NON_TABULAR_EPISODIC_SARSA_SERIAL_BATCH = enum.auto()


class PolicyType(enum.IntEnum):
    # Tabular
    TABULAR_NONE = enum.auto()
    TABULAR_DETERMINISTIC = enum.auto()
    TABULAR_E_GREEDY = enum.auto()
    TABULAR_RANDOM = enum.auto()

    # Non-tabular
    NON_TABULAR_E_GREEDY = enum.auto()
    NON_TABULAR_SOFTMAX_LINEAR = enum.auto()


class DualPolicyRelationship(enum.IntEnum):
    SINGLE_POLICY = enum.auto()
    LINKED_POLICIES = enum.auto()
    INDEPENDENT_POLICIES = enum.auto()


class FeatureType(enum.IntEnum):
    TILE_CODING = enum.auto()


class ValueFunctionType(enum.IntEnum):
    LINEAR_STATE = enum.auto()
    LINEAR_STATE_ACTION = enum.auto()


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
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool
    # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/#what-happens-when-you-start-a-new-process
    # https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn
    FORK_PICKLE = enum.auto()   # copy-on-write copy, with locks but no threads to unlock them (fast, not thread-safe)
    FORK_GLOBAL = enum.auto()   # use a global object (e.g. trainer) to avoid pickling it in args (fastest)
    SPAWN = enum.auto()         # launch fresh interpretters and reload current module, no locks (slower, thread-safe)
    FORK_SERVER = enum.auto()   # have one process be a template and server to spawn copies as small as possible
    #                             thread-safe, smaller memory consumption, unusual, good if doing lots of processes?
