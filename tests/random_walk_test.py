from __future__ import annotations

from mdp import common
from mdp.scenarios.position_move.model import action, state
from mdp.scenarios.random_walk.model.environment import Environment
from mdp.scenarios.random_walk.model.environment_parameters import EnvironmentParameters, default


def random_walk_test() -> bool:
    environment_parameters = EnvironmentParameters(
        environment_type=common.ScenarioType.RANDOM_WALK,
        actions_list=common.ActionsList.NO_ACTIONS
    )
    common.set_none_to_default(environment_parameters, default)
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
