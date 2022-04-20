from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
import utils
from mdp.model.non_tabular.algorithm.batch_mixin.batch__episodic import BatchEpisodic
from mdp.model.non_tabular.value_function.state_action.linear_state_action_shared_weights import \
    LinearStateActionSharedWeights


class BatchSharedWeights(BatchEpisodic, ABC,
                         batch_episodes=common.BatchEpisodes.SHARED_WEIGHTS):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._shared_w: Optional[utils.SharedArrayWrapper] = None

    def attach_to_shared_weights(self, shared_weights_door: utils.SharedArrayDoor):
        self._shared_w = utils.SharedArrayWrapper(shared_array_door=shared_weights_door)
        # TODO: devise a better solution: class variable SharedWeights?
        assert isinstance(self.Q, LinearStateActionSharedWeights)
        self.Q.attach_to_shared_weights(shared_w=self._shared_w)

    def end_episodes(self):
        self._shared_w.close()
