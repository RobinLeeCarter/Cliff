from __future__ import annotations

import common
import scenarios
import view


def view_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    cliff = scenarios.environment_factory(environment_parameters)
    my_view = view.GridView(cliff.grid_world)
    my_view.open_window()
    my_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
