from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    import agent
from algorithm import abstract


class TD0(abstract.EpisodicOnline):
    name: str = "Q-learning"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: dict[str, any]
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._alpha = algorithm_parameters['alpha']
        self.title = f"{TD0.name} α={self._alpha}"

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()

        prev_state = self.agent.prev_state
        reward = self.agent.reward
        state = self.agent.state

        target = reward + self.gamma * self._V[state]
        delta = target - self._V[prev_state]
        self._V[prev_state] += self._alpha * delta
