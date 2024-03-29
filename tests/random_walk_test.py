from __future__ import annotations

from mdp import common
from mdp.task.position_move.model import action, state
from mdp.task.random_walk.model.environment import Environment
from mdp.task.random_walk.model.environment_parameters import EnvironmentParameters


def random_walk_test() -> bool:
    environment_parameters = EnvironmentParameters(
        actions_list=common.ActionsList.NO_ACTIONS
    )
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

    state_ = state.State(is_terminal=False, position=common.XY(x=4, y=0))
    action_ = action.Action(common.XY(x=1, y=0))
    observation_ = environment.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    state_ = state.State(is_terminal=False, position=common.XY(x=5, y=0))
    action_ = action.Action(common.XY(x=1, y=0))
    observation_ = environment.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    state_ = state.State(is_terminal=False, position=common.XY(x=0, y=0))
    action_ = action.Action(common.XY(x=-1, y=0))
    observation_ = environment.from_state_perform_action(state_, action_)
    print(state_, action_)
    print(observation_)

    return True


if __name__ == '__main__':
    if random_walk_test():
        print("Passed")
