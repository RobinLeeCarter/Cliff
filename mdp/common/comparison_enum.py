from __future__ import annotations
import enum


class ComparisonType(enum.IntEnum):
    BLACKJACK_EVALUATION_V = enum.auto()
    BLACKJACK_EVALUATION_Q = enum.auto()
    BLACKJACK_CONTROL_ES = enum.auto()

    CLIFF_ALPHA_START = enum.auto()
    CLIFF_ALPHA_END = enum.auto()
    CLIFF_EPISODE = enum.auto()

    GAMBLER_VALUE_ITERATION_V = enum.auto()

    JACKS_POLICY_ITERATION_V = enum.auto()
    JACKS_VALUE_ITERATION_V = enum.auto()
    JACKS_POLICY_ITERATION_Q = enum.auto()

    RACETRACK_EPISODE = enum.auto()

    RANDOM_WALK_EPISODE = enum.auto()
    WINDY_TIMESTEP = enum.auto()
    WINDY_TIMESTEP_RANDOM = enum.auto()

