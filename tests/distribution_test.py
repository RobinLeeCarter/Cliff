from __future__ import annotations

from mdp.common import Multinoulli
from mdp.task.jacks.model.state import State


x: Multinoulli[State] = Multinoulli()

my_state = State(is_terminal=False, ending_cars_1=5, ending_cars_2=6)
my_state2 = State(is_terminal=False, ending_cars_1=7, ending_cars_2=6)

x[my_state] = 0.8
x[my_state2] += 0.2
print(x[my_state])
print(x[my_state2])

x.enable()

print(x.draw_one())
