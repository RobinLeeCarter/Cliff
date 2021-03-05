from __future__ import annotations
import enum


class Square(enum.IntEnum):
    NORMAL = 0
    CLIFF = 1
    START = 2
    END = 3


class UserEvent(enum.IntEnum):
    NONE = enum.auto()
    QUIT = enum.auto()
    SPACE = enum.auto()


class BreakdownType(enum.IntEnum):
    # NONE = enum.auto()
    EPISODE_BY_TIMESTEP = enum.auto()
    RETURN_BY_EPISODE = enum.auto()
    RMS_BY_EPISODE = enum.auto()
    RETURN_BY_ALPHA = enum.auto()


class AlgorithmType(enum.IntEnum):
    EXPECTED_SARSA = enum.auto()
    VQ = enum.auto()
    Q_LEARNING = enum.auto()
    SARSA = enum.auto()
    CONSTANT_ALPHA_MC = enum.auto()
    TD_0 = enum.auto()
    OFF_POLICY_MC_CONTROL = enum.auto()


class PolicyType(enum.IntEnum):
    DETERMINISTIC = enum.auto()
    E_GREEDY = enum.auto()
    NONE = enum.auto()
    RANDOM = enum.auto()


class DualPolicyRelationship(enum.IntEnum):
    SINGLE_POLICY = enum.auto()
    LINKED_POLICIES = enum.auto()
    INDEPENDENT_POLICIES = enum.auto()


class EnvironmentType(enum.IntEnum):
    CLIFF = enum.auto()
    RANDOM_WALK = enum.auto()
    WINDY = enum.auto()
    RACETRACK = enum.auto()


class ActionsList(enum.IntEnum):
    NO_ACTIONS = enum.auto()
    FOUR_MOVES = enum.auto()
    FOUR_CLIFF_FRIENDLY_MOVES = enum.auto()
    KINGS_MOVES = enum.auto()
    KINGS_MOVES_PLUS_NO_MOVE = enum.auto()


class GridViewType(enum.IntEnum):
    POSITION = enum.auto()
    POSITION_MOVE = enum.auto()
