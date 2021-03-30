from __future__ import annotations

# from typing import Optional

from mdp import common  # , controller, scenario
from mdp.scenarios.factory import scenario_factory
# from mdp.model import model_
# from mdp.view import view_


class Application:
    def __init__(self, comparison_type: common.ComparisonType):
        self._scenario = scenario_factory.scenario_factory(comparison_type)
        self._scenario.build()
        self._scenario.run()

        # self._comparison_type: common.ComparisonType = comparison_type
        # self._scenario_type: common.ScenarioType = common.comparison_to_scenario[comparison_type]

        # self._scenario: Optional[scenario.Scenario] = None

        # self._comparison: common.Comparison = self._scenario._comparison
        #
        # self._model: model_.Model = model_.Model()
        # self._view: view_.View = view_.View()
        # self._controller: controller.Controller = controller.Controller(self._model, self._view)

    # def build(self):
    #

        # enable model and view to send messages to controller
        # self._model.set_controller(self._controller)
        # self._view.set_controller(self._controller)
        #
        # self._controller.build(self._comparison)

    # def run(self):
    #     self._scenario.run()
        # self._controller.run()
