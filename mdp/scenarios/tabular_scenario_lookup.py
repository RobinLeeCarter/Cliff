from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.scenario import Scenario
from mdp import common

from mdp.scenarios.jacks.scenario.jacks_policy_evaluation_q import JacksPolicyEvaluationQ
from mdp.scenarios.jacks.scenario.jacks_policy_evaluation_v import JacksPolicyEvaluationV
from mdp.scenarios.jacks.scenario.jacks_policy_improvement_q import JacksPolicyImprovementQ
from mdp.scenarios.jacks.scenario.jacks_policy_improvement_v import JacksPolicyImprovementV
from mdp.scenarios.jacks.scenario.jacks_policy_iteration_q import JacksPolicyIterationQ
from mdp.scenarios.jacks.scenario.jacks_policy_iteration_v import JacksPolicyIterationV
from mdp.scenarios.jacks.scenario.jacks_value_iteration_q import JacksValueIterationQ
from mdp.scenarios.jacks.scenario.jacks_value_iteration_v import JacksValueIterationV

from mdp.scenarios.blackjack.scenario.blackjack_control_es import BlackjackControlES
from mdp.scenarios.blackjack.scenario.blackjack_evaluation_q import BlackjackEvaluationQ
from mdp.scenarios.blackjack.scenario.blackjack_evaluation_v import BlackjackEvaluationV

from mdp.scenarios.gambler.scenario.gambler_value_iteration_v import GamblerValueIterationV

from mdp.scenarios.racetrack.scenario.racetrack_episode import RacetrackEpisode

from mdp.scenarios.random_walk.scenario.random_walk_episode import RandomWalkEpisode

from mdp.scenarios.cliff.scenario.cliff_alpha_start import CliffAlphaStart
from mdp.scenarios.cliff.scenario.cliff_alpha_end import CliffAlphaEnd
from mdp.scenarios.cliff.scenario.cliff_episode import CliffEpisode

from mdp.scenarios.windy.scenario.windy_timestep import WindyTimestep

lookup_type = dict[common.ComparisonType, Type[Scenario] | tuple[Type[Scenario], dict[str, object]]]


def get_tabular_scenario_lookup() -> lookup_type:
    ct = common.ComparisonType
    lookup: lookup_type = {
        ct.JACKS_POLICY_EVALUATION_Q: JacksPolicyEvaluationQ,
        ct.WINDY_TIMESTEP_RANDOM: (WindyTimestep, {"random_wind": True})
    }
    return lookup


    # # could also consider that this is not encapsulated in scenario
    # # dict could be built by scenario code perhaps
    # # or sceanario could be specified and passed on and then sub-scenario selected
    # # scenario_lookup: dict[common.ComparisonType, BaseScenario]
    # # args_lookup: dict[common.ComparisonType, tuple?, dict?]
    # ct = common.ComparisonType
    # if comparison_type == ct.JACKS_POLICY_EVALUATION_Q:
    #     scenario = JacksPolicyEvaluationQ()
    # elif comparison_type == ct.JACKS_POLICY_EVALUATION_V:
    #     scenario = JacksPolicyEvaluationV()
    # elif comparison_type == ct.JACKS_POLICY_IMPROVEMENT_Q:
    #     scenario = JacksPolicyImprovementQ()
    # elif comparison_type == ct.JACKS_POLICY_IMPROVEMENT_V:
    #     scenario = JacksPolicyImprovementV()
    # elif comparison_type == ct.JACKS_POLICY_ITERATION_Q:
    #     scenario = JacksPolicyIterationQ()
    # elif comparison_type == ct.JACKS_POLICY_ITERATION_V:
    #     scenario = JacksPolicyIterationV()
    # elif comparison_type == ct.JACKS_VALUE_ITERATION_Q:
    #     scenario = JacksValueIterationQ()
    # elif comparison_type == ct.JACKS_VALUE_ITERATION_V:
    #     scenario = JacksValueIterationV()
    #
    # elif comparison_type == ct.BLACKJACK_CONTROL_ES:
    #     scenario = BlackjackControlES()
    # elif comparison_type == ct.BLACKJACK_EVALUATION_Q:
    #     scenario = BlackjackEvaluationQ()
    # elif comparison_type == ct.BLACKJACK_EVALUATION_V:
    #     scenario = BlackjackEvaluationV()
    #
    # elif comparison_type == ct.GAMBLER_VALUE_ITERATION_V:
    #     scenario = GamblerValueIterationV()
    #
    # elif comparison_type == ct.RACETRACK_EPISODE:
    #     scenario = RacetrackEpisode()
    #
    # elif comparison_type == ct.RANDOM_WALK_EPISODE:
    #     scenario = RandomWalkEpisode()
    #
    # elif comparison_type == ct.CLIFF_ALPHA_END:
    #     scenario = CliffAlphaEnd()
    # elif comparison_type == ct.CLIFF_ALPHA_START:
    #     scenario = CliffAlphaStart()
    # elif comparison_type == ct.CLIFF_EPISODE:
    #     scenario = CliffEpisode()
    #
    # elif comparison_type == ct.WINDY_TIMESTEP:
    #     scenario = WindyTimestep(random_wind=False)
    # elif comparison_type == ct.WINDY_TIMESTEP_RANDOM:
    #     scenario = WindyTimestep(random_wind=True)
    #
    # else:
    #     raise ValueError(comparison_type)
    #
    # kwargs: dict = {}
    # if comparison_type == ct.WINDY_TIMESTEP_RANDOM:
    #     kwargs["random_wind"] = True
    #
    # scenario: Scenario = scenario_type(**kwargs)
    #
    # return scenario
