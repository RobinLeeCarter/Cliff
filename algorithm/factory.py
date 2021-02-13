from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    import environment
    import agent
    from algorithm import abstract
import common
from algorithm import control  # , policy_evaluation


class Factory:
    def __init__(self, environment_: environment.Environment, agent_: agent.Agent):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_

    def __getitem__(self, settings_: common.Settings) -> abstract.Episodic:
        a = common.AlgorithmType
        algorithm_lookup: dict[a, Type[abstract.Episodic]] = {
            a.EXPECTED_SARSA: control.ExpectedSarsa,
            a.Q_LEARNING: control.QLearning,
            a.SARSA: control.Sarsa,
            a.VQ: control.VQ
        }
        type_for_algorithm = algorithm_lookup[settings_.algorithm_type]
        algorithm_ = type_for_algorithm(self.environment, self.agent, settings_.algorithm_parameters)
        return algorithm_

        # if settings_.algorithm_type == common.AlgorithmType.Sarsa:
        #     alpha = self.alpha_lookup(settings_)
        #     return control.Sarsa(self.environment, self.agent, alpha, verbose)
        # elif settings_.algorithm_type == common.AlgorithmType.QLearning:
        #     alpha = self.alpha_lookup(settings_)
        #     return control.QLearning(self.environment, self.agent, alpha, verbose)
        # elif settings_.algorithm_type == common.AlgorithmType.ExpectedSarsa:
        #     alpha = self.alpha_lookup(settings_)
        #     return control.ExpectedSarsa(self.environment, self.agent, alpha, verbose)
        # elif settings_.algorithm_type == common.AlgorithmType.VQ:
        #     alpha = self.alpha_lookup(settings_)
        #     alpha_variable = self.alpha_variable_lookup(settings_)
        #     return control.VQ(self.environment, self.agent, alpha, alpha_variable, verbose)

    # def alpha_lookup(self, settings_: common.Settings, default: float = 0.5) -> float:
    #     if "alpha" in settings_.algorithm_parameters:
    #         return settings_.algorithm_parameters["alpha"]
    #     else:
    #         return default
    #
    # def alpha_variable_lookup(self, settings_: common.Settings, default: bool = False) -> bool:
    #     if "alpha_variable" in settings_.algorithm_parameters:
    #         return settings_.algorithm_parameters["alpha_variable"]
    #     else:
    #         return default
