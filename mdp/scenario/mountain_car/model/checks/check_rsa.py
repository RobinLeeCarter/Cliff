from __future__ import annotations
# from typing import TYPE_CHECKING

from mdp.model.non_tabular.agent.rsa import RSA
from mdp.scenario.mountain_car.model.state import State
from mdp.scenario.mountain_car.model.action import Action


my_state = State(is_terminal=False, position=10, velocity=1)
my_action = Action(acceleration=1.0)

my_rsa = RSA[State, Action](r=1.0, state=my_state, action=my_action)

print(type(my_rsa.state))
print(type(my_rsa.action))
