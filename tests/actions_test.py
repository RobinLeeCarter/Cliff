from __future__ import annotations

import utils
from mdp import common
from mdp.scenarios.cliff.model.environment_parameters import EnvironmentParameters \
    as CliffEnvironmentParameters
from mdp.scenarios.cliff.model.environment_parameters import default \
    as cliff_default
from mdp.scenarios.cliff.model.environment import Environment \
    as CliffEnvironment
from mdp.scenarios.random_walk.model.environment_parameters import EnvironmentParameters\
    as RandomWalkEnvironmentParameters
from mdp.scenarios.random_walk.model.environment_parameters import default \
    as random_walk_default
from mdp.scenarios.random_walk.model.environment import Environment \
    as RandomWalkEnvironment


cliff_environment_parameters = CliffEnvironmentParameters(
    environment_type=common.ScenarioType.CLIFF,
    actions_list=common.ActionsList.FOUR_MOVES
)
utils.set_none_to_default(cliff_environment_parameters, cliff_default)
cliff_environment = CliffEnvironment(cliff_environment_parameters)
cliff_environment.build()
print(len(cliff_environment.actions))

random_walk_environment_parameters = RandomWalkEnvironmentParameters(
    environment_type=common.ScenarioType.RANDOM_WALK,
    actions_list=common.ActionsList.NO_ACTIONS
)
utils.set_none_to_default(random_walk_environment_parameters, random_walk_default)
random_walk_environment = RandomWalkEnvironment(random_walk_environment_parameters)
random_walk_environment.build()
print(len(random_walk_environment.actions))
