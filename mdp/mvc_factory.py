from __future__ import annotations
from typing import Type

from mdp.model.general.general_model import GeneralModel
from mdp.view.general.general_view import GeneralView
from mdp.controller.general_controller import GeneralController
from mdp import common

from mdp.scenario.jacks.controller import Controller as JacksController
from mdp.scenario.jacks.model.model import Model as JacksModel
from mdp.scenario.jacks.view.view import View as JacksView

from mdp.scenario.cliff.controller import Controller as CliffController
from mdp.scenario.cliff.model.model import Model as CliffModel
from mdp.scenario.cliff.view.view import View as CliffView

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

MVCLookup = dict[common.EnvironmentType, tuple[Type[GeneralModel], Type[GeneralView], Type[GeneralController]]]


class MVCFactory:
    def __init__(self):
        e = common.EnvironmentType
        self._lookup: MVCLookup = {
            e.JACKS: (JacksModel, JacksView, JacksController),
            e.CLIFF: (CliffModel, CliffView, CliffController),
        }

    def create(self, environment_type: common.EnvironmentType) -> tuple[GeneralModel, GeneralView, GeneralController]:
        # result: Type[Scenario] | tuple[Type[Scenario], dict[str, object]] = self._lookup[scenario_type]

        type_of_model: Type[GeneralModel]
        type_of_view: Type[GeneralView]
        type_of_controller: Type[GeneralController]

        type_of_model, type_of_view, type_of_controller = self._lookup[environment_type]

        model: GeneralModel = type_of_model()
        view: GeneralView = type_of_view()
        controller: GeneralController = type_of_controller()

        return model, view, controller
