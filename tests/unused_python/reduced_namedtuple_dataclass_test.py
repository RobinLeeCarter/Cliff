from dataclasses import dataclass
from collections import namedtuple

XY = namedtuple('XY', ['x', 'y'])


@dataclass(frozen=True)
class Action:
    move: XY
    test: str


my_action = Action(move=XY(3, 4), test="hello")
print(my_action.test)
print(my_action.move.x)
my_action = Action(move=(3, 4), test=56)
print(my_action.test)
print(my_action.move.y)     # fails here

