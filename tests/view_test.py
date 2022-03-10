from __future__ import annotations

from mdp import common
from mdp.scenario.scenario_factory import ScenarioFactory
from mdp.scenario.cliff.scenario.scenario import Scenario
# from mdp.view import view


def view_test() -> bool:
    scenario_factory = ScenarioFactory()
    scenario = scenario_factory.create(common.ComparisonType.CLIFF_EPISODE)
    assert isinstance(scenario, Scenario)
    scenario.build()

    # noinspection PyProtectedMember
    scenario._view.grid_view.display_and_wait()

    # # noinspection PyProtectedMember
    # environment: Environment = scenario._model.environment
    # # noinspection PyProtectedMember
    # comparison: common.Comparison = scenario._comparison
    # my_view = view.View()
    # my_view.build(environment.grid_world, comparison)
    # my_view.grid_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
