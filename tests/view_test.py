from __future__ import annotations

from mdp import common, scenarios
from mdp.scenarios.common.model import environment_factory
from mdp.view import view_


def view_test() -> bool:
    comparison: common.Comparison = scenarios.cliff_episode()
    # environment_parameters = common.EnvironmentParameters(
    #     environment_type=common.EnvironmentType.CLIFF,
    #     actions_list=common.ActionsList.FOUR_MOVES
    # )
    cliff = environment_factory.environment_factory(comparison.environment_parameters)
    my_view = view_.View()
    my_view.build(cliff.grid_world, comparison)
    my_view.grid_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
