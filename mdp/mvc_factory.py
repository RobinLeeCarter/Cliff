from __future__ import annotations
from typing import Type

from mdp.model.general.general_model import GeneralModel
from mdp.view.general.general_view import GeneralView
from mdp.controller.general_controller import GeneralController
from mdp import common

from mdp.scenario.blackjack.model.model import Model as BlackjackModel
from mdp.scenario.blackjack.view.view import View as BlackjackView
from mdp.scenario.blackjack.controller import Controller as BlackjackController

from mdp.scenario.cliff.model.model import Model as CliffModel
from mdp.scenario.cliff.view.view import View as CliffView
from mdp.scenario.cliff.controller import Controller as CliffController

from mdp.scenario.gambler.model.model import Model as GamblerModel
from mdp.scenario.gambler.view.view import View as GamblerView
from mdp.scenario.gambler.controller import Controller as GamblerController

from mdp.scenario.jacks.model.model import Model as JacksModel
from mdp.scenario.jacks.view.view import View as JacksView
from mdp.scenario.jacks.controller import Controller as JacksController

from mdp.scenario.mountain_car.model.model import Model as MountainCarModel
from mdp.scenario.mountain_car.view.view import View as MountainCarView
from mdp.scenario.mountain_car.controller import Controller as MountainCarController

from mdp.scenario.racetrack.model.model import Model as RacetrackModel
from mdp.scenario.racetrack.view.view import View as RacetrackView
from mdp.scenario.racetrack.controller import Controller as RacetrackController

from mdp.scenario.random_walk.model.model import Model as RandomWalkModel
from mdp.scenario.random_walk.view.view import View as RandomWalkView
from mdp.scenario.random_walk.controller import Controller as RandomWalkController

from mdp.scenario.windy.model.model import Model as WindyModel
from mdp.scenario.windy.view.view import View as WindyView
from mdp.scenario.windy.controller import Controller as WindyController

MVCLookup = dict[common.EnvironmentType, tuple[Type[GeneralModel], Type[GeneralView], Type[GeneralController]]]


class MVCFactory:
    def __init__(self):
        e = common.EnvironmentType
        self._lookup: MVCLookup = {
            e.BLACKJACK: (BlackjackModel, BlackjackView, BlackjackController),
            e.CLIFF: (CliffModel, CliffView, CliffController),
            e.GAMBLER: (GamblerModel, GamblerView, GamblerController),
            e.JACKS: (JacksModel, JacksView, JacksController),
            e.MOUNTAIN_CAR: (MountainCarModel, MountainCarView, MountainCarController),
            e.RACETRACK: (RacetrackModel, RacetrackView, RacetrackController),
            e.RANDOM_WALK: (RandomWalkModel, RandomWalkView, RandomWalkController),
            e.WINDY: (WindyModel, WindyView, WindyController),
        }

    def create(self, environment_type: common.EnvironmentType) -> tuple[GeneralModel, GeneralView, GeneralController]:
        type_of_model: Type[GeneralModel]
        type_of_view: Type[GeneralView]
        type_of_controller: Type[GeneralController]

        type_of_model, type_of_view, type_of_controller = self._lookup[environment_type]

        model: GeneralModel = type_of_model()
        view: GeneralView = type_of_view()
        controller: GeneralController = type_of_controller()

        return model, view, controller
