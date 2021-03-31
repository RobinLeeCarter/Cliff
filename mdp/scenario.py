from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from mdp import common
from mdp.model.model import Model
from mdp.view.view import View
from mdp.controller import Controller


class Scenario(ABC):
    def __init__(self, comparison_type: common.ComparisonType):
        self._comparison_type: common.ComparisonType = comparison_type
        self._comparison: Optional[common.Comparison] = None

        # self._model: Optional[Model] = None
        # self._view: Optional[View] = None
        # self._controller: Optional[Controller] = None

        self._model: Model = self._create_model()
        self._view: View = self._create_view()
        self._controller: Controller = self._get_controller()
        # self._model.set_controller(self._controller)
        # self._view.set_controller(self._controller)
        self._controller.link_mvc(self._model, self._view)

    def build(self):
        self._set_comparison()
        # self._comparison = self._get_comparison(self._comparison_type)
        self._controller.build(self._comparison)

    def run(self):
        self._controller.run()
        self._controller.output()

    def _create_model(self) -> Model:
        return Model()

    def _create_view(self) -> View:
        return View()

    def _get_controller(self) -> Controller:
        return Controller()

    @abstractmethod
    def _set_comparison(self):
        pass

    # @abstractmethod
    # def _get_comparison(self, comparison_type: common.ComparisonType) -> common.Comparison:
    #     pass

