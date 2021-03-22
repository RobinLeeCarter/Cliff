from __future__ import annotations

from mdp.common import Distribution
from mdp.scenarios.jacks.state import State


x: Distribution[State] = Distribution()

my_state = State(is_terminal=False, cars_cob_1=5, cars_cob_2=6)
my_state2 = State(is_terminal=False, cars_cob_1=7, cars_cob_2=6)

x[my_state] = 0.8
x[my_state2] += 0.2
print(x[my_state])
print(x[my_state2])

x.self_test()



