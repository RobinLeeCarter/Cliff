from __future__ import annotations

from mdp import common
from mdp.scenario import Scenario
from mdp.scenarios import scenario_factory


def get_scenario(comparison_type: common.ComparisonType, do_build: bool = True) -> Scenario:
    scenario = scenario_factory.scenario_factory(comparison_type)
    if do_build:
        scenario.build()
    return scenario
