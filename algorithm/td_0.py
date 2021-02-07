import constants
import environment
import agent
from algorithm import abstract


class TD0(abstract.EpisodicOnline):
    name: str = "Q-learning"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        super().__init__(environment_, agent_, verbose)
        self.title = f"{TD0.name} α={alpha}"
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()

        prev_state = self.agent.prev_state
        reward = self.agent.reward
        state = self.agent.state

        target = reward + constants.GAMMA * self._V[state]
        delta = target - self._V[prev_state]
        self._V[prev_state] += self._alpha * delta
