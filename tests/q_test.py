from __future__ import annotations

from mdp import common
from mdp.model.algorithm.value_function import state_action_function
from mdp.scenarios.common.model import environment_factory_
from mdp.scenarios.common.model.position_move import action, state


def q_test() -> bool:
    environment_parameters = common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    environment_ = environment_factory_.environment_factory(environment_parameters)
    q = state_action_function.StateActionFunction(environment_, initial_q_value=-7.0)

    state_ = state.State(is_terminal=False, position=common.XY(x=4, y=2))
    state_index = environment_.state_index[state_]
    print(f"state_.index {state_index}")

    action_ = action.Action(common.XY(x=1, y=0))
    action_index = environment_.action_index[action_]
    print(f"action_.index {action_index}")

    print(q[state_, action_])
    q[state_, action_] = 2.0
    q[state_, action_] += 0.5
    print(q[state_, action_])

    # noinspection PyProtectedMember
    print(f"q {q._values}")

    return True


if __name__ == '__main__':
    if q_test():
        print("Passed")
