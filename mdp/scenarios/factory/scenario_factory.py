from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.scenario import Scenario as BaseScenario
    from mdp.model.environment.environment import Environment as BaseEnvironment
from mdp import common

from mdp.scenarios.jacks.scenario import Scenario as JacksScenario

from mdp.scenarios.cliff.environment import Environment as CliffEnvironment
from mdp.scenarios.random_walk.environment import Environment as RandomWalkEnvironment
from mdp.scenarios.windy.environment import Environment as WindyEnvironment
from mdp.scenarios.racetrack.environment import Environment as RacetrackEnvironment

from mdp.scenarios.blackjack.environment import Environment as BlackjackEnvironment
from mdp.scenarios.gambler.environment import Environment as GamblerEnvironment


def scenario_factory(comparison_type: common.ComparisonType) -> BaseScenario:
    scenario_type = common.comparison_to_scenario[comparison_type]
    st = common.ScenarioType
    if scenario_type == st.JACKS:
        scenario = JacksScenario(comparison_type, scenario_type)
    # elif environment_type == et.RANDOM_WALK:
    #     environment_ = RandomWalkEnvironment(environment_parameters)
    # elif environment_type == et.WINDY:
    #     environment_ = WindyEnvironment(environment_parameters)
    # elif environment_type == et.RACETRACK:
    #     environment_ = RacetrackEnvironment(environment_parameters)
    # elif environment_type == et.JACKS:
    #     environment_ = JacksEnvironment(environment_parameters)
    # elif environment_type == et.BLACKJACK:
    #     environment_ = BlackjackEnvironment(environment_parameters)
    # elif environment_type == et.GAMBLER:
    #     environment_ = GamblerEnvironment(environment_parameters)
    else:
        raise ValueError(scenario_type)
    # environment_.build()
    return scenario
