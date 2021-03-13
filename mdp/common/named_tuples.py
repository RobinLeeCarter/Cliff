from __future__ import annotations
from collections import namedtuple
from typing import NamedTuple


XY = namedtuple('XY', ['x', 'y'])


class RGBA(NamedTuple):
    r: int
    g: int
    b: int
    a: int
