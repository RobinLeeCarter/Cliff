from __future__ import annotations
# from typing import TYPE_CHECKING

from mdp.common import Distribution
from mdp import common
from mdp.scenarios import scenario_factory

from mdp.scenarios.jacks.model.state import State
from mdp.scenarios.jacks.model.action import Action
from mdp.scenarios.jacks.model.environment import Environment
from mdp.scenarios.jacks.model.dynamics.dynamics import Dynamics

Response = tuple[float, State]


scenario = scenario_factory.scenario_factory(common.ComparisonType.JACKS_POLICY_ITERATION_V)
scenario.build()
# noinspection PyProtectedMember
environment: Environment = scenario._model.environment

# comparison = jacks_policy_iteration_v()
# environment_ = unused_environment_factory.environment_factory(comparison.environment_parameters)

# dynamics = Dynamics(environment_, comparison.environment_parameters)
# dynamics.build()
dynamics: Dynamics = environment.dynamics

state = State(is_terminal=False, ending_cars_1=10, ending_cars_2=8)
action = Action(transfer_1_to_2=2)

summary_outcomes: Distribution[Response] = dynamics.get_summary_outcomes(state, action)
all_outcomes: Distribution[Response] = dynamics.get_all_outcomes(state, action)

# compresses to 6,468 outcomes
print(len(summary_outcomes.dict))
print(len(all_outcomes.dict))

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
