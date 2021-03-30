from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from mdp import common
from mdp.model.model_ import Model
from mdp.view.view_ import View
from mdp.controller import Controller


class Scenario(ABC):
    def __init__(self, comparison_type: common.ComparisonType, scenario_type: common.ScenarioType):
        self.comparison_type: common.ComparisonType = comparison_type
        self.scenario_type: common.ScenarioType = scenario_type
        self.comparison: Optional[common.Comparison] = None

        self.model: Optional[Model] = None
        self.view: Optional[View] = None
        self.controller: Optional[Controller] = None

    def build(self):
        self.comparison = self._get_comparison(self.comparison_type)
        self.model = self.get_model()
        self.view = self.get_view()
        self.controller = self.get_controller()

    def get_model(self) -> Model:
        return Model()

    def get_view(self) -> View:
        return View()

    def get_controller(self) -> Controller:
        return Controller(self.model, self.view)

    @abstractmethod
    def _get_comparison(self, comparison_type: common.ComparisonType) -> common.Comparison:
        pass
