from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.scenario.general_scenario import GeneralScenario
    from mdp.model.general.general_model import GeneralModel
    from mdp.view.general.general_view import GeneralView
    from mdp.controller.general_controller import GeneralController
from mdp.scenario.scenario_factory import ScenarioFactory
from mdp.mvc_factory import MVCFactory


class Application:
    def __init__(self, scenario_type: common.ScenarioType):
        self._scenario_factory: ScenarioFactory = ScenarioFactory()
        # self._mvc_factory: MVCFactory = MVCFactory()
        #
        # self._comparison: common.Comparison = self._scenario_factory.create(scenario_type)
        #
        # self._model: GeneralModel
        # self._view: GeneralView
        # self._controller: GeneralController
        # environment_type: common.EnvironmentType = self._comparison.environment_parameters.environment_type
        # self._model, self._view, self._controller = self._mvc_factory.create(environment_type)

        self._scenario: GeneralScenario = self._scenario_factory.create(scenario_type)
        self._scenario.build()
        self._scenario.run()

    # def _build(self):
    #     self._comparison: common.Comparison = self._create_comparison()
    #     # self._comparison = self._get_comparison(self._scenario_type)
    #     self._controller.build(self._comparison)
    #
    # def _run(self):
    #     self._controller.run()
    #     self._controller.output()
