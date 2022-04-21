from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import math
import os
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import itertools

import numpy as np

import utils
from utils import SharedArrayWrapper, SharedArrayDoor
from mdp.model.non_tabular.algorithm.batch_mixin.batch__episodic import BatchEpisodic
from mdp.model.non_tabular.algorithm.batch_mixin.batch_delta_weights import BatchDeltaWeights
from mdp.model.non_tabular.algorithm.batch_mixin.batch_feature_trajectories import BatchFeatureTrajectories
from mdp.model.non_tabular.algorithm.batch_mixin.batch_trajectories import BatchTrajectories


if TYPE_CHECKING:
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder
from mdp import common

_trainer: Trainer
_door: SharedArrayDoor


class ParallelEpisodesSharedWeights:
    def __init__(self, trainer: Trainer):
        self._trainer: Trainer = trainer
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        self._settings = self._trainer.settings
        # settings.result_parameters.return_cum_timestep = True
        self._parallel_context_type: Optional[common.ParallelContextType] = self._settings.episode_multiprocessing
        self._processes: int = min(os.cpu_count(), self._settings.episodes_per_batch)
        self._episodes_per_process: int = int(math.ceil(self._settings.episodes_per_batch / self._processes))
        self._actual_episodes_per_batch: int = self._processes * self._episodes_per_process

        self._results: list[common.Result] = []
        self._recorder: Optional[Recorder] = None
        if self._trainer.breakdown:
            self._recorder = self._trainer.breakdown.recorder

        # if self._parallel_context_type is None it should fail here
        context_str = common.parallel_context_str[self._parallel_context_type]
        self._context: mp.context.BaseContext = mp.get_context(context_str)
        self._use_global_trainer: bool = (self._parallel_context_type == common.ParallelContextType.FORK_GLOBAL)
        if self._use_global_trainer:
            global _trainer
            _trainer = self._trainer

        self._weights: Optional[np.ndarray] = None

    @property
    def actual_episodes_per_batch(self) -> int:
        return self._actual_episodes_per_batch

    def set_weights(self, weights: np.ndarray):
        self._weights = weights

    def do_episode_batch(self, starting_episode: int):
        seeds: list[int] = utils.Rng.get_seeds(number_of_seeds=self._processes)
        episode_counter_starts: list[int] = \
            [starting_episode + x
             for x in range(0, self._actual_episodes_per_batch, self._episodes_per_process)]
        episodes_to_do: list[int] = list(itertools.repeat(self._episodes_per_process, self._processes))
        result_parameter_list: list[common.ResultParameters] = self._get_result_parameter_list()

        algorithm = self._trainer.algorithm
        assert isinstance(algorithm, BatchEpisodic)
        algorithm.start_episodes()

        # TODO: possibly don't want to do this for each batch
        # TODO: make generic enough to be combined with parallel_episodes.py
        with SharedMemoryManager() as shared_memory_manager:
            shared_weights: SharedArrayWrapper = \
                SharedArrayWrapper().build(self._context, shared_memory_manager, source=self._weights)

            with self._context.Pool(processes=self._processes,
                                    initializer=_init,
                                    initargs=(shared_weights.door, )) as pool:
                # TODO: pass seeds in init_args also
                if self._use_global_trainer:
                    args = zip(seeds,
                               episode_counter_starts,
                               episodes_to_do,
                               result_parameter_list)
                    self._results = pool.starmap(_global_do_episodes_wrapper, args)
                else:
                    args = zip(itertools.repeat(self._trainer),
                               seeds,
                               episode_counter_starts,
                               episodes_to_do,
                               result_parameter_list)
                    self._results = pool.starmap(_do_episodes_wrapper, args)
            shared_weights.copy_back()

        self._unpack_results()

        # the agent is already set up in trainer.trainer so just apply the final result to it
        # TODO: should this be commented out or pass in function?
        # self._trainer.algorithm.apply_result(result=self._results[-1])

    def _get_result_parameter_list(self) -> list[common.ResultParameters]:
        rp_norm: common.ResultParameters = common.ResultParameters(
            return_recorder=True,
        )
        result_parameter_list: list[common.ResultParameters] = list(itertools.repeat(rp_norm, self._processes))
        return result_parameter_list

    def _unpack_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown so is ready to be used for output
        if self._trainer.breakdown:
            unique_recorders = set(result.recorder for result in self._results)
            for recorder in unique_recorders:
                self._recorder.add_recorder(recorder)

        algorithm = self._trainer.algorithm
        match algorithm.batch_episodes:
            case common.BatchEpisodes.DELTA_WEIGHTS:
                assert isinstance(algorithm, BatchDeltaWeights)
                delta_w_vectors = [result.delta_w_vector for result in self._results]
                algorithm.apply_delta_w_vectors(delta_w_vectors)
            case common.BatchEpisodes.TRAJECTORIES:
                assert isinstance(algorithm, BatchTrajectories)
                for result in self._results:
                    algorithm.add_trajectories(result.trajectories)
                algorithm.apply_trajectories()
            case common.BatchEpisodes.FEATURE_TRAJECTORIES:
                assert isinstance(algorithm, BatchFeatureTrajectories)
                for result in self._results:
                    algorithm.add_feature_trajectories(result.feature_trajectories)
                algorithm.apply_feature_trajectories()

        # self._trainer.max_cum_timestep = max(result.cum_timestep for result in self._results)


def _init(door: SharedArrayDoor):
    global _door
    _door = door


def _global_do_episodes_wrapper(seed: int,
                                episode_counter_start: int,
                                episodes_to_do: int,
                                result_parameters: common.ResultParameters)\
        -> common.Result:
    global _door
    utils.Rng.set_seed(seed)
    return _trainer.do_episodes(episode_counter_start=episode_counter_start,
                                episodes_to_do=episodes_to_do,
                                shared_weights_door=_door,
                                result_parameters=result_parameters)


def _do_episodes_wrapper(trainer: Trainer,
                         seed: int,
                         episode_counter_start: int,
                         episodes_to_do: int,
                         result_parameters: common.ResultParameters)\
        -> common.Result:
    global _door
    utils.Rng.set_seed(seed)
    return trainer.do_episodes(episode_counter_start=episode_counter_start,
                               episodes_to_do=episodes_to_do,
                               shared_weights_door=_door,
                               result_parameters=result_parameters)


# def _train_map_wrapper(train_tuple: tuple[Trainer, common.Settings]) -> common.Result:
#     # created so that chucksize can be set in map
#     trainer, settings = train_tuple
#     return trainer.train(settings)
