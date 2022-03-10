from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
    from mdp.scenario.general_scenario import GeneralScenario
from mdp.scenario.scenario_factory import ScenarioFactory


class Application:
    def __init__(self, scenario_type: common.ScenarioType):
        self._scenario_factory: ScenarioFactory = ScenarioFactory()
        self._scenario: GeneralScenario = self._scenario_factory.create(scenario_type)
        self._scenario.build()
        self._scenario.run()
