from __future__ import annotations

from typing import TypeVar
from mdp.scenarios.jacks.state import State


T = TypeVar('T')


class Distribution(dict[T, float]):
    def __missing__(self, key) -> float:
        return 0.0


# x: Distribution[State] = {}
x: Distribution[State] = Distribution()

my_state = State(is_terminal=False, cars_cob_1=5, cars_cob_2=6)
my_state2 = State(is_terminal=False, cars_cob_1=7, cars_cob_2=6)

x[my_state] = "fish"
x[my_state2] += 8.8
x["banana"] = 7.7
print(x[my_state])
print(x[my_state2])



