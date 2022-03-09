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

ScenarioLookup = dict[common.ComparisonType, Type[Scenario] | tuple[Type[Scenario], dict[str, object]]]


class ScenarioFactory:
    def __init__(self):
        ct = common.ComparisonType
        self._lookup: ScenarioLookup = {
            ct.JACKS_POLICY_EVALUATION_Q: JacksPolicyEvaluationQ,
            ct.JACKS_POLICY_EVALUATION_V: JacksPolicyEvaluationV,
            ct.JACKS_POLICY_IMPROVEMENT_Q: JacksPolicyImprovementQ,
            ct.JACKS_POLICY_IMPROVEMENT_V: JacksPolicyImprovementV,
            ct.JACKS_POLICY_ITERATION_Q: JacksPolicyIterationQ,
            ct.JACKS_POLICY_ITERATION_V: JacksPolicyIterationV,
            ct.JACKS_VALUE_ITERATION_Q: JacksValueIterationQ,
            ct.JACKS_VALUE_ITERATION_V: JacksValueIterationV,
            ct.BLACKJACK_CONTROL_ES: BlackjackControlES,
            ct.BLACKJACK_EVALUATION_Q: BlackjackEvaluationQ,
            ct.BLACKJACK_EVALUATION_V: BlackjackEvaluationV,
            ct.GAMBLER_VALUE_ITERATION_V: GamblerValueIterationV,
            ct.RACETRACK_EPISODE: RacetrackEpisode,
            ct.RANDOM_WALK_EPISODE: RandomWalkEpisode,
            ct.CLIFF_ALPHA_END: CliffAlphaEnd,
            ct.CLIFF_ALPHA_START: CliffAlphaStart,
            ct.CLIFF_EPISODE: CliffEpisode,
            ct.WINDY_TIMESTEP: WindyTimestep,
            ct.WINDY_TIMESTEP_RANDOM: (WindyTimestep, {"random_wind": True})
        }

    def create(self, comparison_type: common.ComparisonType) -> Scenario:
        # result: Type[Scenario] | tuple[Type[Scenario], dict[str, object]] = self._lookup[comparison_type]
        scenario_type: Type[Scenario]
        kwargs: dict[str, object] = {}

        match self._lookup[comparison_type]:
            case type() as scenario_type:
                pass
            case type() as scenario_type, dict() as kwargs:
                pass
            case _:
                raise Exception("scenario type / args lookup failed")

        scenario: Scenario = scenario_type(**kwargs)
        return scenario


# def scenario_factory(comparison_type: common.ComparisonType) -> Scenario:
#     # could also consider that this is not encapsulated in scenario
#     # dict could be built by scenario code perhaps
#     # or sceanario could be specified and passed on and then sub-scenario selected
#     # scenario_lookup: dict[common.ComparisonType, BaseScenario]
#     # args_lookup: dict[common.ComparisonType, tuple?, dict?]
#     ct = common.ComparisonType
#     scenario_type: Type[Scenario] = Scenario
#
#     kwargs: dict = {}
#     if comparison_type == ct.WINDY_TIMESTEP_RANDOM:
#         kwargs["random_wind"] = True
#
#     scenario: Scenario = scenario_type(**kwargs)
#
#     return scenario
