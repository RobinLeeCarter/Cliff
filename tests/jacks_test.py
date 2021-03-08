from __future__ import annotations

from mdp import common
from mdp.scenarios.factory import environment_factory
from mdp.scenarios.jacks import action, state, comparisons


def jacks_test() -> bool:
    comparison: common.Comparison = comparisons.jacks_policy_iteration()
    environment_ = environment_factory.environment_factory(comparison.environment_parameters)

    print("States...")
    for state_ in environment_.states:
        print(f"{state_}")
    print()

    print("Actions...")
    for action_ in environment_.actions:
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
