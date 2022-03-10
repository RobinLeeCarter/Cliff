from __future__ import annotations
from typing import Type

from mdp.scenario.general_scenario import GeneralScenario
from mdp import common

from mdp.scenario.jacks.scenario.jacks_policy_evaluation_q import JacksPolicyEvaluationQ
from mdp.scenario.jacks.scenario.jacks_policy_evaluation_v import JacksPolicyEvaluationV
from mdp.scenario.jacks.scenario.jacks_policy_improvement_q import JacksPolicyImprovementQ
from mdp.scenario.jacks.scenario.jacks_policy_improvement_v import JacksPolicyImprovementV
from mdp.scenario.jacks.scenario.jacks_policy_iteration_q import JacksPolicyIterationQ
from mdp.scenario.jacks.scenario.jacks_policy_iteration_v import JacksPolicyIterationV
from mdp.scenario.jacks.scenario.jacks_value_iteration_q import JacksValueIterationQ
from mdp.scenario.jacks.scenario.jacks_value_iteration_v import JacksValueIterationV

from mdp.scenario.blackjack.scenario.blackjack_control_es import BlackjackControlES
from mdp.scenario.blackjack.scenario.blackjack_evaluation_q import BlackjackEvaluationQ
from mdp.scenario.blackjack.scenario.blackjack_evaluation_v import BlackjackEvaluationV

from mdp.scenario.gambler.scenario.gambler_value_iteration_v import GamblerValueIterationV

from mdp.scenario.racetrack.scenario.racetrack_episode import RacetrackEpisode

from mdp.scenario.random_walk.scenario.random_walk_episode import RandomWalkEpisode

from mdp.scenario.cliff.scenario.cliff_alpha_start import CliffAlphaStart
from mdp.scenario.cliff.scenario.cliff_alpha_end import CliffAlphaEnd
from mdp.scenario.cliff.scenario.cliff_episode import CliffEpisode

from mdp.scenario.windy.scenario.windy_timestep import WindyTimestep

ScenarioLookup = dict[common.ScenarioType, Type[GeneralScenario] | tuple[Type[GeneralScenario], dict[str, any]]]


class ScenarioFactory:
    def __init__(self):
        ct = common.ScenarioType
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

    def create(self, scenario_type: common.ScenarioType) -> GeneralScenario:
        # result: Type[Scenario] | tuple[Type[Scenario], dict[str, object]] = self._lookup[scenario_type]
        type_of_scenario: Type[GeneralScenario]
        kwargs: dict[str, any] = {}

        match self._lookup[scenario_type]:
            case type() as type_of_scenario:
                pass
            case type() as type_of_scenario, dict() as kwargs:
                pass
            case _:
                raise Exception("scenario type / args lookup failed")

        scenario: GeneralScenario = type_of_scenario(**kwargs)
        return scenario
