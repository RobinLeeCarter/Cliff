from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment as BaseEnvironment
from mdp import common
from mdp.scenarios.cliff.model.environment import Environment as CliffEnvironment
from mdp.scenarios.random_walk.model.environment import Environment as RandomWalkEnvironment
from mdp.scenarios.windy.environment import Environment as WindyEnvironment
from mdp.scenarios.racetrack.model.environment import Environment as RacetrackEnvironment
from mdp.scenarios.jacks.model.environment import Environment as JacksEnvironment
from mdp.scenarios.blackjack.model.environment import Environment as BlackjackEnvironment
from mdp.scenarios.gambler.model.environment import Environment as GamblerEnvironment


def environment_factory(environment_parameters: common.EnvironmentParameters) -> BaseEnvironment:
    environment_type = environment_parameters.environment_type
    et = common.ScenarioType

    if environment_type == et.CLIFF:
        environment_ = CliffEnvironment(environment_parameters)
    elif environment_type == et.RANDOM_WALK:
        environment_ = RandomWalkEnvironment(environment_parameters)
    elif environment_type == et.WINDY:
        environment_ = WindyEnvironment(environment_parameters)
    elif environment_type == et.RACETRACK:
        environment_ = RacetrackEnvironment(environment_parameters)
    elif environment_type == et.JACKS:
        environment_ = JacksEnvironment(environment_parameters)
    elif environment_type == et.BLACKJACK:
        environment_ = BlackjackEnvironment(environment_parameters)
    elif environment_type == et.GAMBLER:
        environment_ = GamblerEnvironment(environment_parameters)
    else:
        raise NotImplementedError
    environment_.build()
    return environment_
