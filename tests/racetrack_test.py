from __future__ import annotations

import utils
from mdp import common
from mdp.scenarios.racetrack.model import action, grids, state
from mdp.scenarios.racetrack.model.environment import Environment
from mdp.scenarios.racetrack.model.environment_parameters import EnvironmentParameters, default


def racetrack_test() -> bool:
    environment_parameters = EnvironmentParameters(
        grid=grids.TRACK_1
    )
    utils.set_none_to_default(environment_parameters, default)
    environment = Environment(environment_parameters)
    environment.build()

    for state_ in environment.states:
        state_index = environment.state_index[state_]
        print(f"{state_} \t index={state_index}")

    print()

    for action_ in environment.actions:
        action_index = environment.action_index[action_]
        print(f"{action_} \t index={action_index}")

    print()

    state_ = state.State(is_terminal=False, position=common.XY(x=4, y=0), velocity=common.XY(x=0, y=1))
    action_ = action.Action(acceleration=common.XY(x=1, y=0))
    response_ = environment.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    state_ = state.State(is_terminal=False, position=common.XY(x=5, y=4), velocity=common.XY(x=1, y=0))
    action_ = action.Action(acceleration=common.XY(x=0, y=0))
    response_ = environment.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    state_ = state.State(is_terminal=False, position=common.XY(x=0, y=0), velocity=common.XY(x=0, y=3))
    action_ = action.Action(common.XY(x=0, y=-1))
    response_ = environment.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(response_)

    return True


if __name__ == '__main__':
    if racetrack_test():
        print("Passed")
