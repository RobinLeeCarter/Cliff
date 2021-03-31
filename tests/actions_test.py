from __future__ import annotations

from mdp import common
from unused import unused_environment_factory

environment_parameters = common.EnvironmentParameters(
    environment_type=common.ScenarioType.CLIFF,
    actions_list=common.ActionsList.FOUR_MOVES
)
cliff = unused_environment_factory.environment_factory(environment_parameters)
print(len(cliff.actions))

environment_parameters = common.EnvironmentParameters(
    environment_type=common.ScenarioType.RANDOM_WALK,
    actions_list=common.ActionsList.NO_ACTIONS
)
random_walk = unused_environment_factory.environment_factory(environment_parameters)
print(len(random_walk.actions))

print(len(cliff.actions))
