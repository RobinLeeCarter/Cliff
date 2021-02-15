from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    import environment
    import agent
    from algorithm import abstract
import common
from algorithm import control  # , policy_evaluation


def factory(environment_: environment.Environment,
            agent_: agent.Agent,
            algorithm_parameters: common.Settings.algorithm_parameters) -> abstract.Episodic:
    a = common.AlgorithmType
    algorithm_lookup: dict[a, Type[abstract.Episodic]] = {
        a.EXPECTED_SARSA: control.ExpectedSarsa,
        a.Q_LEARNING: control.QLearning,
        a.SARSA: control.Sarsa,
        a.VQ: control.VQ
    }
    type_for_algorithm = algorithm_lookup[algorithm_parameters.algorithm_type]
    algorithm_ = type_for_algorithm(environment_, agent_, algorithm_parameters)
    return algorithm_
