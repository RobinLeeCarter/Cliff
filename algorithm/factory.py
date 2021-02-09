from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
    from comparison.comparison_dataclasses import settings

from algorithm import abstract, control  # , policy_evaluation


class Factory:
    def __init__(self, environment_: environment.Environment, agent_: agent.Agent):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_

    def __getitem__(self, settings_: settings.Settings) -> abstract.Episodic:
        if "verbose" in settings_.parameters:
            verbose: bool = settings_.parameters["verbose"]
        else:
            verbose: bool = False

        if settings_.algorithm_type == control.Sarsa:
            alpha = self.alpha_lookup(settings_)
            return control.Sarsa(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == control.QLearning:
            alpha = self.alpha_lookup(settings_)
            return control.QLearning(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == control.ExpectedSarsa:
            alpha = self.alpha_lookup(settings_)
            return control.ExpectedSarsa(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == control.VQ:
            alpha = self.alpha_lookup(settings_)
            alpha_variable = self.alpha_variable_lookup(settings_)
            return control.VQ(self.environment, self.agent, alpha, alpha_variable, verbose)

    def alpha_lookup(self, settings_: settings.Settings, default: float = 0.5) -> float:
        if "alpha" in settings_.parameters:
            return settings_.parameters["alpha"]
        else:
            return default

    def alpha_variable_lookup(self, settings_: settings.Settings, default: bool = False) -> bool:
        if "alpha_variable" in settings_.parameters:
            return settings_.parameters["alpha_variable"]
        else:
            return default
