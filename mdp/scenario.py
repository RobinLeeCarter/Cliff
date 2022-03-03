from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.general.environment import Environment
from mdp import common
from mdp.model.model import Model
from mdp.view.view import View
from mdp.controller import Controller


class Scenario(ABC):
    def __init__(self):
        # self._comparison_type: common.ComparisonType = comparison_type
        self._model: Model = self._create_model()
        self._view: View = self._create_view()
        self._controller: Controller = self._create_controller()
        self._controller.link_mvc(self._model, self._view)
        self._comparison: Optional[common.Comparison] = None

    def build(self):
        self._comparison: common.Comparison = self._create_comparison()
        # self._comparison = self._get_comparison(self._comparison_type)
        self._controller.build(self._comparison)

    def run(self):
        self._controller.run()
        self._controller.output()

    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _create_controller(self) -> Controller:
        return Controller()

    @abstractmethod
    def _create_comparison(self) -> common.Comparison:
        pass

    @property
    def environment(self) -> Environment:
        environment = self._model.environment
        if environment is None:
            raise Exception("Environment is None")
        else:
            return self._model.environment

    # @abstractmethod
    # def _get_comparison(self, comparison_type: common.ComparisonType) -> common.Comparison:
    #     pass

