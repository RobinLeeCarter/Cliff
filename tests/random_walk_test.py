from __future__ import annotations

import common
import environment
import scenarios


def random_walk_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.RANDOM_WALK,
        actions_list=common.ActionsList.STATIONARY_MOVE_ONLY
    )
    environment_ = scenarios.environment_factory(environment_parameters)

    for state_ in environment_.states():
        print(f"{state_} \t index={state_.index}")

    print()

    for action_ in environment_.actions():
        print(f"{action_} \t index={action_.index}")

    print()

    state_ = environment.State(common.XY(x=4, y=0))
    action_ = environment.Action(common.XY(x=1, y=0))
    observation_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    state_ = environment.State(common.XY(x=5, y=0))
    action_ = environment.Action(common.XY(x=1, y=0))
    observation_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    state_ = environment.State(common.XY(x=0, y=0))
    action_ = environment.Action(common.XY(x=-1, y=0))
    observation_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    return True


if __name__ == '__main__':
    if random_walk_test():
        print("Passed")
