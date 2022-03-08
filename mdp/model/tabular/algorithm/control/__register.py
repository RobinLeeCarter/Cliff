from typing import Callable, Type

from mdp.model.tabular.algorithm.abstract.algorithm import Algorithm
from mdp.model.tabular.algorithm.control.dp_policy_iteration_q import DpPolicyIterationQ
from mdp.model.tabular.algorithm.control.dp_policy_iteration_v import DpPolicyIterationV
from mdp.model.tabular.algorithm.control.dp_value_iteration_q_deterministic import DpValueIterationQ
from mdp.model.tabular.algorithm.control.dp_value_iteration_v_deterministic import DpValueIterationV
from mdp.model.tabular.algorithm.control.mc_control_on_policy import McControlOnPolicy
from mdp.model.tabular.algorithm.control.mc_control_off_policy import McControlOffPolicy
from mdp.model.tabular.algorithm.control.vq import VQ
from mdp.model.tabular.algorithm.control.expected_sarsa import ExpectedSarsa
from mdp.model.tabular.algorithm.control.sarsa import Sarsa
from mdp.model.tabular.algorithm.control.q_learning import QLearning


def register_control_algorithms(register: Callable[[Type[Algorithm]], None]):
    algorithms = [DpPolicyIterationQ,
                  DpPolicyIterationV,
                  DpValueIterationQ,
                  DpValueIterationV,
                  McControlOnPolicy,
                  McControlOffPolicy,
                  VQ,
                  ExpectedSarsa,
                  Sarsa,
                  QLearning]
    for algorithm in algorithms:
        register(algorithm)
