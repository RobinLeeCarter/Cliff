import environment
import agent
from algorithm import settings, episodic_algorithm, sarsa_alg, q_learning, vq, expected_sarsa


class Factory:
    def __init__(self, environment_: environment.Environment, agent_: agent.Agent):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_

    def __getitem__(self, settings_: settings.Settings) -> episodic_algorithm.EpisodicAlgorithm:
        if "verbose" in settings_.parameters:
            verbose: bool = settings_.parameters["verbose"]
        else:
            verbose: bool = False

        if settings_.algorithm_type == sarsa_alg.SarsaAlg:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return sarsa_alg.SarsaAlg(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == q_learning.QLearning:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return q_learning.QLearning(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == expected_sarsa.ExpectedSarsa:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return expected_sarsa.ExpectedSarsa(self.environment, self.agent, alpha, verbose)
        elif settings_.algorithm_type == vq.VQ:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            if "alpha_variable" in settings_.parameters:
                alpha_variable: bool = True
            else:
                alpha_variable: bool = False
            return vq.VQ(self.environment, self.agent, alpha, alpha_variable, verbose)
