from __future__ import annotations

from mdp import common
from mdp.scenarios.factory import environment_factory
from mdp.scenarios.blackjack import comparisons  # action, state,


def blackjack_test() -> bool:
    comparison: common.Comparison = comparisons.blackjack_comparison_v()
    environment_ = environment_factory.environment_factory(comparison.environment_parameters)

    print("States...")
    for state_ in environment_.states:
        print(f"{state_}")
    print()

    print("Actions...")
    for action_ in environment_.actions:
        print(f"{action_}")
    print()

    print(f"total states = {len(environment_.states)}")
    print(f"total actions = {len(environment_.actions)}")

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
    if blackjack_test():
        print("Passed")
