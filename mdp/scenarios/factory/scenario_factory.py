from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario import Scenario as BaseScenario
from mdp import common

from mdp.scenarios.jacks.scenario.jacks_policy_iteration_v import JacksPolicyIterationV
from mdp.scenarios.jacks.scenario.jacks_policy_iteration_q import JacksPolicyIterationQ
from mdp.scenarios.jacks.scenario.jacks_value_iteration_v import JacksValueIterationV


def scenario_factory(comparison_type: common.ComparisonType) -> BaseScenario:
    ct = common.ComparisonType
    if comparison_type == ct.JACKS_POLICY_ITERATION_V:
        scenario = JacksPolicyIterationV(comparison_type)
    elif comparison_type == ct.JACKS_POLICY_ITERATION_Q:
        scenario = JacksPolicyIterationQ(comparison_type)
    elif comparison_type == ct.JACKS_VALUE_ITERATION_V:
        scenario = JacksValueIterationV(comparison_type)

    # elif comparison_type == ct.JACKS_VALUE_ITERATION_V:
    #     scenario = JacksValueIterationV(comparison_type)

    # elif environment_type == et.RANDOM_WALK:
    #     environment_ = RandomWalkEnvironment(environment_parameters)
    # elif environment_type == et.WINDY:
    #     environment_ = WindyEnvironment(environment_parameters)
    # elif environment_type == et.RACETRACK:
    #     environment_ = RacetrackEnvironment(environment_parameters)
    # elif environment_type == et.JACKS:
    #     environment_ = JacksEnvironment(environment_parameters)
    # elif environment_type == et.BLACKJACK:
    #     environment_ = BlackjackEnvironment(environment_parameters)
    # elif environment_type == et.GAMBLER:
    #     environment_ = GamblerEnvironment(environment_parameters)
    else:
        raise ValueError(scenario_type)
    # environment_.build()
    return scenario
