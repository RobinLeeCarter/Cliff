from typing import Callable

# from mdp.model.tabular.algorithm.control import dp_policy_iteration_q
from mdp.model.tabular.algorithm.control.dp_policy_iteration_q import DpPolicyIterationQ


# def register2(register_callback: Callable):
#     dp_policy_iteration_q.register2(register_callback)


def register4(register_callback: Callable):
    register_callback(DpPolicyIterationQ)
    ...
