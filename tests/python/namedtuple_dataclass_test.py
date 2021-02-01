# from __future__ import annotations

from dataclasses import dataclass
from collections import namedtuple


XY = namedtuple('XY', ['x', 'y'])


@dataclass(frozen=True)
class Action:
    move: XY


def dataclass_test() -> bool:
    my_xy = XY(x=2, y=3)
    print(my_xy.x)
    my_action = Action(move=XY(3, 4))
    print(my_action.move.x)
    my_action = Action(move=(3, 4))
    print(my_action.move.x)
    return True


if __name__ == '__main__':
    if dataclass_test():
        print("Passed")
