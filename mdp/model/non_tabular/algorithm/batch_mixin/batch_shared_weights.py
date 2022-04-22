from __future__ import annotations
from abc import ABC
from multiprocessing import Lock
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
import utils
from mdp.model.non_tabular.algorithm.batch_mixin.batch__episodic import BatchEpisodic
from mdp.model.non_tabular.value_function.state_action.linear_state_action_function import LinearStateActionFunction


class BatchSharedWeights(BatchEpisodic, ABC,
                         batch_episodes=common.BatchEpisodes.SHARED_WEIGHTS,
                         store_feature_vectors=True):
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._shared_w: Optional[utils.SharedArrayWrapper] = None
        self._w_lock: Optional[Lock] = None

    def attach_to_shared_weights(self, shared_weights_door: utils.SharedArrayDoor):
        self._shared_w = utils.SharedArrayWrapper().attach(shared_weights_door)
        self._w_lock: Lock = self._shared_w.lock
        # TODO: use self.Q.has_w
        assert isinstance(self.Q, LinearStateActionFunction)
        self.Q.w = self._shared_w.array
        # if self.Q.shared_weights:
        #     assert isinstance(self.Q, LinearStateActionSharedWeights)
        #     self.Q.attach_to_shared_weights(shared_w=self._shared_w)

    def end_episodes(self):
        self._shared_w.close()
