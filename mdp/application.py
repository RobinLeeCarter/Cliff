from __future__ import annotations

from mdp import common
from mdp.scenarios import scenario_factory


class Application:
    def __init__(self, comparison_type: common.ComparisonType):
        self._scenario = scenario_factory.scenario_factory(comparison_type)
        self._scenario.build()
        self._scenario.run()
