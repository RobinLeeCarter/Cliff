from __future__ import annotations

from mdp import common
from mdp.model import scenarios
from mdp.model.scenarios.common import action_move, state_position


def random_walk_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.RANDOM_WALK,
        actions_list=common.ActionsList.NO_ACTIONS
    )
    environment_ = scenarios.environment_factory(environment_parameters)

    for state_ in environment_.states:
        state_index = environment_.state_index[state_]
        print(f"{state_} \t index={state_index}")

    print()

    for action_ in environment_.actions:
        action_index = environment_.action_index[action_]
        print(f"{action_} \t index={action_index}")

    print()

    state_ = state_position.State(is_terminal=False, position=common.XY(x=4, y=0))
    action_ = action_move.Action(common.XY(x=1, y=0))
    observation_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    state_ = state_position.State(is_terminal=False, position=common.XY(x=5, y=0))
    action_ = action_move.Action(common.XY(x=1, y=0))
    observation_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    state_ = state_position.State(is_terminal=False, position=common.XY(x=0, y=0))
    action_ = action_move.Action(common.XY(x=-1, y=0))
    observation_ = environment_.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    return True


if __name__ == '__main__':
    if random_walk_test():
        print("Passed")
