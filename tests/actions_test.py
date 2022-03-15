from __future__ import annotations

from mdp import common
from mdp.scenario.cliff.comparison.environment_parameters import EnvironmentParameters \
    as CliffEnvironmentParameters
from mdp.scenario.cliff.model.environment import Environment \
    as CliffEnvironment
from mdp.scenario.random_walk.comparison.environment_parameters import EnvironmentParameters\
    as RandomWalkEnvironmentParameters
from mdp.scenario.random_walk.model.environment import Environment \
    as RandomWalkEnvironment


cliff_environment_parameters = CliffEnvironmentParameters(
    actions_list=common.ActionsList.FOUR_MOVES
)
cliff_environment = CliffEnvironment(cliff_environment_parameters)
cliff_environment.build()
print(len(cliff_environment.actions))

random_walk_environment_parameters = RandomWalkEnvironmentParameters(
    actions_list=common.ActionsList.NO_ACTIONS
)
random_walk_environment = RandomWalkEnvironment(random_walk_environment_parameters)
random_walk_environment.build()
print(len(random_walk_environment.actions))
