from __future__ import annotations

from mdp import common
from mdp.scenarios import scenario_factory
# from mdp.scenarios.cliff.model.environment import Environment
# from mdp.view import view


def view_test() -> bool:
    scenario = scenario_factory.scenario_factory(common.ComparisonType.CLIFF_EPISODE)
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
