from __future__ import annotations
# from typing import TYPE_CHECKING

from mdp.scenarios.jacks.state import State
from mdp.scenarios.jacks.action import Action
from mdp.scenarios.jacks.response import Response
from mdp.scenarios.jacks.comparisons import jacks_policy_evaluation
from mdp.scenarios.jacks.dynamics.dynamics import Dynamics
from mdp.scenarios.jacks.dynamics.outcome import Outcome

comparison = jacks_policy_evaluation()

dynamics = Dynamics(comparison.environment_parameters)
dynamics.build()

state = State(is_terminal=False, cars_cob_1=10, cars_cob_2=8)
action = Action(transfer_1_to_2=2)

summary_outcomes: list[Outcome] = dynamics.get_summary_outcomes(state, action)
all_outcomes: list[Outcome] = dynamics.get_all_outcomes(state, action)

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
