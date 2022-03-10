from __future__ import annotations

from mdp import common
from mdp.scenario.scenario_factory import ScenarioFactory
from mdp.scenario.jacks.scenario.scenario import Scenario
# from mdp.scenarios.jacks.model.state import State
# from mdp.scenarios.jacks.model.action import Action
# from mdp.scenarios.jacks.model.response import Response
from mdp.scenario.jacks.model.environment import Environment


def jacks_test() -> bool:
    scenario_factory = ScenarioFactory()
    scenario = scenario_factory.create(common.ComparisonType.JACKS_POLICY_ITERATION_V)
    assert isinstance(scenario, Scenario)
    scenario.build()

    # noinspection PyProtectedMember
    environment: Environment = scenario._model.environment
    assert isinstance(environment, Environment)

    print("States...")
    for state in environment.states:
        print(f"{state}")
    print()

    print("Actions...")
    for action in environment.actions:
        print(f"{action}")
    print()

    # state_ = state.State(is_terminal=False, position=common.XY(x=4, y=0))
    # action_ = action.Action(common.XY(x=1, y=0))
    # observation_ = environment_.from_state_perform_action(state_, action_)
    # print(state_, action_)
    # print(observation_)
    #
    # state_ = state.State(is_terminal=False, position=common.XY(x=5, y=0))
    # action_ = action.Action(common.XY(x=1, y=0))
    # observation_ = environment_.from_state_perform_action(state_, action_)
    # print(state_, action_)
    # print(observation_)
    #
    # state_ = state.State(is_terminal=False, position=common.XY(x=0, y=0))
    # action_ = action.Action(common.XY(x=-1, y=0))
    # observation_ = environment_.from_state_perform_action(state_, action_)
    # print(state_, action_)
    # print(observation_)

    return True


if __name__ == '__main__':
    if jacks_test():
        print("Passed")
