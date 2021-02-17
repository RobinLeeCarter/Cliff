from __future__ import annotations

from mdp import common
from mdp.model import scenarios
from mdp import view


def view_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    cliff = scenarios.environment_factory(environment_parameters)
    my_view = view.View()
    my_view.build(cliff.grid_world)
    my_view.grid_view.open_window()
    my_view.grid_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
