from __future__ import annotations
from typing import Type

from mdp.model.base.base_model import BaseModel
from mdp.view.base.base_view import BaseView
from mdp.controller.base_controller import BaseController
from mdp import common

from mdp.task.blackjack.model.model import Model as BlackjackModel
from mdp.task.blackjack.view.view import View as BlackjackView
from mdp.task.blackjack.controller import Controller as BlackjackController

from mdp.task.cliff.model.model import Model as CliffModel
from mdp.task.cliff.view.view import View as CliffView
from mdp.task.cliff.controller import Controller as CliffController

from mdp.task.gambler.model.model import Model as GamblerModel
from mdp.task.gambler.view.view import View as GamblerView
from mdp.task.gambler.controller import Controller as GamblerController

from mdp.task.jacks.model.model import Model as JacksModel
from mdp.task.jacks.view.view import View as JacksView
from mdp.task.jacks.controller import Controller as JacksController

from mdp.task.mountain_car.model.model import Model as MountainCarModel
from mdp.task.mountain_car.view.view import View as MountainCarView
from mdp.task.mountain_car.controller import Controller as MountainCarController

from mdp.task.racetrack.model.model import Model as RacetrackModel
from mdp.task.racetrack.view.view import View as RacetrackView
from mdp.task.racetrack.controller import Controller as RacetrackController

from mdp.task.random_walk.model.model import Model as RandomWalkModel
from mdp.task.random_walk.view.view import View as RandomWalkView
from mdp.task.random_walk.controller import Controller as RandomWalkController

from mdp.task.windy.model.model import Model as WindyModel
from mdp.task.windy.view.view import View as WindyView
from mdp.task.windy.controller import Controller as WindyController

MVCLookup = dict[common.EnvironmentType, tuple[Type[BaseModel], Type[BaseView], Type[BaseController]]]


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

    def create(self, environment_type: common.EnvironmentType) -> tuple[BaseModel, BaseView, BaseController]:
        type_of_model: Type[BaseModel]
        type_of_view: Type[BaseView]
        type_of_controller: Type[BaseController]

        type_of_model, type_of_view, type_of_controller = self._lookup[environment_type]

        model: BaseModel = type_of_model()
        view: BaseView = type_of_view()
        controller: BaseController = type_of_controller()

        return model, view, controller
