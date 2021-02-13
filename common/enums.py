from __future__ import annotations
import enum


class Square(enum.IntEnum):
    NORMAL = 0
    CLIFF = 1
    START = 2
    END = 3
    AGENT = 4


class UserEvent(enum.IntEnum):
    NONE = enum.auto()
    QUIT = enum.auto()
    SPACE = enum.auto()


class ComparisonType(enum.IntEnum):
    EPISODE_BY_TIMESTEP = enum.auto()
    RETURN_BY_EPISODE = enum.auto()
    RETURN_BY_ALPHA = enum.auto()


class AlgorithmType(enum.IntEnum):
    EXPECTED_SARSA = enum.auto()
    VQ = enum.auto()
    Q_LEARNING = enum.auto()
    SARSA = enum.auto()


class EnvironmentType(enum.IntEnum):
    CLIFF = enum.auto()
    RANDOM_WALK = enum.auto()
    WINDY = enum.auto()
