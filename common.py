import enum
from collections import namedtuple
# from dataclasses import dataclass


XY = namedtuple('XY', ['x', 'y'])

#
# @dataclass(frozen=True)
# class XY:
#     x: int
#     y: int


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


class Comparison(enum.IntEnum):
    RETURN_BY_EPISODE = enum.auto()
    RETURN_BY_ALPHA = enum.auto()
