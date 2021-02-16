from __future__ import annotations

import common
from mdp.model import scenarios

environment_parameters = common.EnvironmentParameters(
    environment_type=common.EnvironmentType.CLIFF,
    actions_list=common.ActionsList.FOUR_MOVES
)
cliff = scenarios.environment_factory(environment_parameters)
print(cliff.actions_shape)

environment_parameters = common.EnvironmentParameters(
    environment_type=common.EnvironmentType.RANDOM_WALK,
    actions_list=common.ActionsList.STATIONARY_MOVE_ONLY
)
random_walk = scenarios.environment_factory(environment_parameters)
print(random_walk.actions_shape)

print(cliff.actions_shape)
