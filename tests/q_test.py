from __future__ import annotations

from mdp import common
from mdp.model.tabular.value_function import state_action_function
from mdp.task.cliff.model.environment_parameters import EnvironmentParameters
from mdp.task.cliff.model.environment import Environment
from mdp.task._position_move.model.action import Action
from mdp.task._position_move.model.state import State


def q_test() -> bool:
    environment_parameters = EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    environment = Environment(environment_parameters)
    environment.build()
    q = state_action_function.StateActionFunction(environment, initial_value=-7.0)

    state_ = State(is_terminal=False, position=common.XY(x=4, y=2))
    s = environment.state_index[state_]
    print(f"state_.index {s}")

    action_ = Action(common.XY(x=1, y=0))
    a = environment.action_index[action_]
    print(f"action_.index {a}")

    print(q[s, a])
    q[s, a] = 2.0
    q[s, a] += 0.5
    print(q[s, a])

    # noinspection PyProtectedMember
    print(f"Q: {q.matrix}")

    return True


if __name__ == '__main__':
    if q_test():
        print("Passed")
