from __future__ import annotations
# from typing import TYPE_CHECKING

from mdp.common import Distribution

from mdp.scenarios.jacks.model.state import State
from mdp.scenarios.jacks.model.action import Action
from mdp.scenarios.jacks.model.response import Response
from mdp.scenarios.jacks.scenario.scenario import jacks_policy_iteration_v
# from mdp.scenarios.jacks.dynamics.dynamics import Dynamics

from unused import unused_environment_factory

comparison = jacks_policy_iteration_v()
environment_ = unused_environment_factory.environment_factory(comparison.environment_parameters)

# dynamics = Dynamics(environment_, comparison.environment_parameters)
# dynamics.build()
dynamics = environment_.dynamics

state = State(is_terminal=False, ending_cars_1=10, ending_cars_2=8)
action = Action(transfer_1_to_2=2)

summary_outcomes: Distribution[Response] = dynamics.get_summary_outcomes(state, action)
all_outcomes: Distribution[Response] = dynamics.get_all_outcomes(state, action)

# compresses to 6,468 outcomes
print(len(summary_outcomes))
print(len(all_outcomes))

for outcome1, outcome2 in zip(summary_outcomes, all_outcomes):
    print(outcome1)
    print(outcome2)

print("-----")

for outcome in all_outcomes:
    print(outcome)

response: Response = dynamics.draw_response(state, action)
print(response)
response: Response = dynamics.draw_response(state, action)
print(response)
response: Response = dynamics.draw_response(state, action)
print(response)
response: Response = dynamics.draw_response(state, action)
print(response)
response: Response = dynamics.draw_response(state, action)
print(response)


# for outcome in outcomes:
#     if outcome.probability > 0.001:
#         print(outcome.state)
#         print(outcome.reward)
#         print(outcome.probability)
