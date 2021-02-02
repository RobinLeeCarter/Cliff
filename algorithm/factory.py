import environment
import agent
from algorithm import settings, episodic_algorithm, sarsa


class Factory:
    def __init__(self, environment_: environment.Environment, agent_: agent.Agent):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_

    def __getitem__(self, settings_: settings.Settings) -> episodic_algorithm.EpisodicAlgorithm:
        if "verbose" in settings_.parameters:
            verbose: bool = settings_.parameters["verbose"]
        else:
            verbose: bool = False

        if settings_.algorithm_type == sarsa.Sarsa:
            if "alpha" in settings_.parameters:
                alpha: float = settings_.parameters["alpha"]
            else:
                alpha: float = 0.5
            return sarsa.Sarsa(self.environment, self.agent, alpha, verbose)
