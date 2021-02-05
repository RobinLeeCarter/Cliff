import environment
import agent
from comparison import settings
from algorithm import algorithms
# from algorithm import vq, q_learning, episodic_algorithm, expected_sarsa, sarsa_alg


class Factory:
    def __init__(self, environment_: environment.Environment, agent_: agent.Agent):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_

    def __getitem__(self, settings_: settings.Settings) -> algorithms.EpisodicAlgorithm:
        if "verbose" in settings_.parameters:
            verbose: bool = settings_.parameters["verbose"]
        else:
            verbose: bool = False

        if settings_.algorithm_type == algorithms.SarsaAlg:
            alpha = self.alpha_lookup(settings_)
            return algorithms.SarsaAlg(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == algorithms.QLearning:
            alpha = self.alpha_lookup(settings_)
            return algorithms.QLearning(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == algorithms.ExpectedSarsa:
            alpha = self.alpha_lookup(settings_)
            return algorithms.ExpectedSarsa(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == algorithms.VQ:
            alpha = self.alpha_lookup(settings_)
            alpha_variable = self.alpha_variable_lookup(settings_)
            return algorithms.VQ(self.environment, self.agent, alpha, alpha_variable, verbose)

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
