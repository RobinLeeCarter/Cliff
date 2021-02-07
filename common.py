import numpy as np

import enum
from collections import namedtuple
# from dataclasses import dataclass


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


XY = namedtuple('XY', ['x', 'y'])

# @dataclass(frozen=True)
# class XY:
#     x: int
#     y: int

rng: np.random.Generator = np.random.default_rng()
COMPARISON: ComparisonType = ComparisonType.RETURN_BY_ALPHA
