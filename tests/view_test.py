from __future__ import annotations

from mdp import common
from mdp.application import Application
from mdp.scenario.cliff.view.view import View


def view_test() -> bool:
    application = Application(common.ComparisonType.CLIFF_EPISODE)
    # comparison_factory = ComparisonFactory()
    # scenario = comparison_factory.create(common.ComparisonType.CLIFF_EPISODE)
    # assert isinstance(scenario, ComparisonBuilder)
    # scenario.build()

    view = application.view
    assert isinstance(view, View)
    view.grid_view.display_and_wait()

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
