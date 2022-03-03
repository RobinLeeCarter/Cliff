from __future__ import annotations

from mdp import common
from mdp.scenarios import scenario_factory

# from mdp.scenarios.jacks.model.state import State
# from mdp.scenarios.jacks.model.action import Action
# from mdp.scenarios.jacks.model.response import Response
from mdp.scenarios.jacks.model.environment import Environment


def jacks_test() -> bool:
    scenario = scenario_factory.scenario_factory(common.ComparisonType.JACKS_POLICY_ITERATION_V)
    scenario.build()

    environment: Environment = scenario.environment     # type: ignore
    assert isinstance(environment, Environment)

    print("States...")
    for state_ in environment.states:
        print(f"{state_}")
    print()

    print("Actions...")
    for action_ in environment.actions:
        print(f"{action_}")
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
