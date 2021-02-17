from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment
from mdp import common
from mdp.model.scenarios import cliff, windy, random_walk


def environment_factory(environment_parameters: common.EnvironmentParameters) -> environment.Environment:
    environment_type = environment_parameters.environment_type
    et = common.EnvironmentType

    if environment_type == et.CLIFF:
        environment_ = cliff.environment.Environment(environment_parameters)
    elif environment_type == et.WINDY:
        environment_ = windy.environment.Environment(environment_parameters)
    elif environment_type == et.RANDOM_WALK:
        environment_ = random_walk.environment.Environment(environment_parameters)
    else:
        raise NotImplementedError
    return environment_
