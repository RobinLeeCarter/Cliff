from __future__ import annotations

from mdp import common
from mdp.model.algorithm.value_function import state_action_function
from mdp.scenarios.cliff.model.environment_parameters import default
from mdp.scenarios.cliff.model.environment_parameters import EnvironmentParameters
from mdp.scenarios.cliff.model.environment import Environment
from mdp.scenarios.position_move.model import action, state


def q_test() -> bool:
    environment_parameters = EnvironmentParameters(
        environment_type=common.ScenarioType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    )
    common.set_none_to_default(environment_parameters, default)
    environment = Environment(environment_parameters)
    environment.build()
    q = state_action_function.StateActionFunction(environment, initial_q_value=-7.0)

    state_ = state.State(is_terminal=False, position=common.XY(x=4, y=2))
    state_index = environment.state_index[state_]
    print(f"state_.index {state_index}")

    action_ = action.Action(common.XY(x=1, y=0))
    action_index = environment.action_index[action_]
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
