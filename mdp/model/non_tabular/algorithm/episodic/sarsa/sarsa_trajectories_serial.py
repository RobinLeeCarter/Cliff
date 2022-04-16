from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp import common
from mdp.model.non_tabular.value_function.state_action.linear_state_action_function import LinearStateActionFunction
from mdp.model.non_tabular.algorithm.episodic.sarsa.sarsa_trajectories import SarsaTrajectories


class SarsaTrajectoriesSerial(SarsaTrajectories,
                              algorithm_type=common.AlgorithmType.NT_SARSA_TRAJECTORIES_SERIAL,
                              algorithm_name="Sarsa Serial Trajectories"):
    """requires Q be a LinearStateActionFunction"""
    def __init__(self,
                 agent: NonTabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)
        self._w_copy: Optional[np.ndarray] = None

    def start_episodes(self):
        super().start_episodes()
        assert isinstance(self.Q, LinearStateActionFunction)
        # take a copy so can start again
        self._w_copy = self.Q.w.copy()

    def end_episodes(self):
        # immediately reapply episodes
        self._apply_trajectories()

    def _apply_trajectories(self):
        """for use with batch episodes but a single process"""
        # copy back original w and reapply episodes
        assert isinstance(self.Q, LinearStateActionFunction)
        # start again with previous w
        self.Q.w = self._w_copy
        # then apply each trajectory as normal
        super().apply_trajectories()
