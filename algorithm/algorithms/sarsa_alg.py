import constants
import environment
import agent
from algorithm.algorithms import episodic_algorithm


class SarsaAlg(episodic_algorithm.EpisodicAlgorithm):
    name: str = "Sarsa"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        title = f"{SarsaAlg.name} α={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def _start_episode(self):
        self.agent.choose_action()

    def _do_training_step(self):
        self.agent.take_action()
        self.agent.choose_action()
        sarsa = self.agent.get_sarsa()
        delta = sarsa.reward \
            + constants.GAMMA * self._Q[sarsa.state, sarsa.action] \
            - self._Q[sarsa.prev_state, sarsa.prev_action]
        self._Q[sarsa.prev_state, sarsa.prev_action] += self._alpha * delta
        self.agent.policy[sarsa.prev_state] = self._Q.argmax_over_actions(sarsa.prev_state)
