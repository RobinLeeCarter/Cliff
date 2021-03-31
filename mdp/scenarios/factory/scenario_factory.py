from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario import Scenario as BaseScenario
from mdp import common

from mdp.scenarios.jacks.scenario.jacks_policy_iteration_v import JacksPolicyIterationV
from mdp.scenarios.jacks.scenario.jacks_policy_iteration_q import JacksPolicyIterationQ
from mdp.scenarios.jacks.scenario.jacks_value_iteration_v import JacksValueIterationV

from mdp.scenarios.blackjack.scenario.blackjack_control_es import BlackjackControlES
from mdp.scenarios.blackjack.scenario.blackjack_evaluation_q import BlackjackEvaluationQ
from mdp.scenarios.blackjack.scenario.blackjack_evaluation_v import BlackjackEvaluationV

from mdp.scenarios.gambler.scenario.gambler_value_iteration_v import GamblerValueIterationV


def scenario_factory(comparison_type: common.ComparisonType) -> BaseScenario:
    ct = common.ComparisonType
    if comparison_type == ct.JACKS_POLICY_ITERATION_V:
        scenario = JacksPolicyIterationV()
    elif comparison_type == ct.JACKS_POLICY_ITERATION_Q:
        scenario = JacksPolicyIterationQ()
    elif comparison_type == ct.JACKS_VALUE_ITERATION_V:
        scenario = JacksValueIterationV()

    elif comparison_type == ct.BLACKJACK_CONTROL_ES:
        scenario = BlackjackControlES()
    elif comparison_type == ct.BLACKJACK_EVALUATION_Q:
        scenario = BlackjackEvaluationQ()
    elif comparison_type == ct.BLACKJACK_EVALUATION_V:
        scenario = BlackjackEvaluationV()

    elif comparison_type == ct.GAMBLER_VALUE_ITERATION_V:
        scenario = GamblerValueIterationV()

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
