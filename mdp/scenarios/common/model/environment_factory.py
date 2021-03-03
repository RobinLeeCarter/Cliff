from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment
from mdp import common
from mdp.scenarios.cliff.environment import CliffEnvironment
from mdp.scenarios.random_walk.environment import RandomWalkEnvironment
from mdp.scenarios.windy.environment import WindyEnvironment


def factory(environment_parameters: common.EnvironmentParameters) -> environment.Environment:
    environment_type = environment_parameters.environment_type
    et = common.EnvironmentType

    if environment_type == et.CLIFF:
        environment_ = CliffEnvironment(environment_parameters)
    elif environment_type == et.RANDOM_WALK:
        environment_ = RandomWalkEnvironment(environment_parameters)
    elif environment_type == et.WINDY:
        environment_ = WindyEnvironment(environment_parameters)
    else:
        raise NotImplementedError
    return environment_
