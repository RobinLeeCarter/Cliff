import environment
import agent
from algorithm import algorithms, settings
# from algorithm. import vq, q_learning, episodic_algorithm, expected_sarsa, sarsa_alg


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
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return algorithms.SarsaAlg(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == algorithms.QLearning:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return algorithms.QLearning(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == algorithms.ExpectedSarsa:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return algorithms.ExpectedSarsa(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == algorithms.VQ:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            if "alpha_variable" in settings_.parameters:
                alpha_variable: bool = True
            else:
                alpha_variable: bool = False
            return algorithms.VQ(self.environment, self.agent, alpha, alpha_variable, verbose)
