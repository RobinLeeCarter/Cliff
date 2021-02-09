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
    RETURN_BY_EPISODE = enum.auto()
    RETURN_BY_ALPHA = enum.auto()
