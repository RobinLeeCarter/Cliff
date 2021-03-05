from __future__ import annotations

from mdp import common
from mdp.scenarios.factory import environment_factory

environment_parameters = common.EnvironmentParameters(
    environment_type=common.EnvironmentType.CLIFF,
    actions_list=common.ActionsList.FOUR_MOVES
)
cliff = environment_factory.environment_factory(environment_parameters)
print(len(cliff.actions))

environment_parameters = common.EnvironmentParameters(
    environment_type=common.EnvironmentType.RANDOM_WALK,
    actions_list=common.ActionsList.NO_ACTIONS
)
random_walk = environment_factory.environment_factory(environment_parameters)
print(len(random_walk.actions))

print(len(cliff.actions))
