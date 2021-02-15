from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    import environment
    import agent
    from algorithm import abstract
import common
from algorithm import control  # , policy_evaluation


# TODO: Make this consistent
class Factory:
    def __init__(self, environment_: environment.Environment, agent_: agent.Agent):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_

    def __getitem__(self, algorithm_parameters: common.Settings.algorithm_parameters) -> abstract.Episodic:
        a = common.AlgorithmType
        algorithm_lookup: dict[a, Type[abstract.Episodic]] = {
            a.EXPECTED_SARSA: control.ExpectedSarsa,
            a.Q_LEARNING: control.QLearning,
            a.SARSA: control.Sarsa,
            a.VQ: control.VQ
        }
        type_for_algorithm = algorithm_lookup[algorithm_parameters.algorithm_type]
        algorithm_ = type_for_algorithm(self.environment, self.agent, algorithm_parameters)
        return algorithm_
