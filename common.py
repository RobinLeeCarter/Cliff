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
    OBSTACLE = 1
    START = 2
    END = 3
    AGENT = 4


class UserEvent(enum.IntEnum):
    NONE = 0
    QUIT = 1
    SPACE = 2
