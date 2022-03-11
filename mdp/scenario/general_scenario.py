from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Generic, TypeVar

from mdp import common
from mdp.model.general.general_model import GeneralModel
from mdp.view.tabular.tabular_view import TabularView
from mdp.controller.tabular_controller import TabularController


Model = TypeVar('Model', bound=GeneralModel)
View = TypeVar('View', bound=TabularView)
Controller = TypeVar('Controller', bound=TabularController)


class GeneralScenario(Generic[Model, View, Controller], ABC):
    def __init__(self, **_ignored):
        # self._scenario_type: common.ScenarioType = scenario_type
        self._model: Model = self._create_model()
        self._view: View = self._create_view()
        self._controller: Controller = self._create_controller()
        self._controller.link_mvc(self._model, self._view)
        self._comparison: Optional[common.Comparison] = None

    def build(self):
        self._comparison: common.Comparison = self._create_comparison()
        # self._comparison = self._get_comparison(self._scenario_type)
        self._controller.build(self._comparison)

    def run(self):
        self._controller.run()
        self._controller.output()

    @abstractmethod
    def _create_model(self) -> Model:
        ...

    @abstractmethod
    def _create_view(self) -> View:
        ...

    @abstractmethod
    def _create_controller(self) -> Controller:
        ...

    @abstractmethod
    def _create_comparison(self) -> common.Comparison:
        ...
