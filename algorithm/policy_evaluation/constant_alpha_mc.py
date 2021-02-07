import environment
import agent
from algorithm import abstract


class ConstantAlphaMC(abstract.EpisodicMonteCarlo):
    name: str = "Constant-α MC"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        super().__init__(environment_, agent_, verbose)
        self.title = f"{ConstantAlphaMC.name} α={alpha}"
        self._alpha = alpha

    def _process_time_step(self, t: int):
        state = self.episode[t].state
        target = self.episode.G[t]
        delta = target - self._V[state]
        self._V[state] += self._alpha * delta
