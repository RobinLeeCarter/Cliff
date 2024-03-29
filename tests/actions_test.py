from __future__ import annotations

from mdp import common
from mdp.task.cliff.model.environment_parameters import EnvironmentParameters \
    as CliffEnvironmentParameters
from mdp.task.cliff.model.environment import Environment \
    as CliffEnvironment
from mdp.task.random_walk.model.environment_parameters import EnvironmentParameters\
    as RandomWalkEnvironmentParameters
from mdp.task.random_walk.model.environment import Environment \
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
