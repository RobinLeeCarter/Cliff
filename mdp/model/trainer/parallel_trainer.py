from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import multiprocessing as mp
import itertools

import utils
from mdp import common

if TYPE_CHECKING:
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder

_trainer: Trainer


class ParallelTrainer:
    def __init__(self, trainer: Trainer, parallel_context_type: common.ParallelContextType):
        self._trainer: Trainer = trainer
        self._parallel_context_type: common.ParallelContextType = parallel_context_type
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        self._settings_list: list[common.Settings] = []
        self._results: list[common.Result] = []
        self._recorder: Optional[Recorder] = None
        if self._trainer.breakdown:
            self._recorder = self._trainer.breakdown.recorder

        context_str = common.parallel_context_str[self._parallel_context_type]
        self._ctx: mp.context.BaseContext = mp.get_context(context_str)
        self._use_global_trainer: bool = (self._parallel_context_type == common.ParallelContextType.FORK_GLOBAL)
        if self._use_global_trainer:
            global _trainer
            _trainer = self._trainer

    def train(self, settings_list: list[common.Settings]):
        self._settings_list = settings_list
        seeds: list[int] = utils.Rng.get_seeds(number_of_seeds=len(self._settings_list))

        # have final settings return everything (if used in case of V and Q)
        self._set_result_parameters()

        with self._ctx.Pool() as pool:
            if self._use_global_trainer:
                args = zip(seeds, self._settings_list)
                self._results = pool.starmap(_global_train_wrapper, args)
            else:
                args = zip(itertools.repeat(self._trainer), seeds, self._settings_list)
                # self._results = pool.map(_train_map_wrapper, args)
                self._results = pool.starmap(_train_starmap_wrapper, args)

        self._unpack_results()

        # set up agent using final setting and apply the final result
        self._trainer.apply_settings(settings=self._settings_list[-1])
        self._trainer.algorithm.apply_result(result=self._results[-1])

    def _set_result_parameters(self):
        final_settings = self._settings_list[-1]
        for settings in self._settings_list:
            rp: common.ResultParameters = common.ResultParameters(
                return_recorder=True,
            )
            if settings == final_settings:
                rp.return_policy_vector = True
                rp.return_v_vector = True   # will return only if exists
                rp.return_q_matrix = True   # will return only if exists
            settings.result_parameters = rp

    def _unpack_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown so is ready to be used for output
        if self._trainer.breakdown:
            unique_recorders = set(result.recorder for result in self._results)
            for recorder in unique_recorders:
                self._recorder.add_recorder(recorder)


# Avoids pickling trainer?
def _global_train_wrapper(seed: int, settings: common.Settings) -> common.Result:
    utils.Rng.set_seed(seed)
    return _trainer.train(settings, return_result=True)


def _train_starmap_wrapper(trainer: Trainer, seed: int, settings: common.Settings) -> common.Result:
    utils.Rng.set_seed(seed)
    return trainer.train(settings, return_result=True)


# def _train_map_wrapper(train_tuple: tuple[Trainer, common.Settings]) -> common.Result:
#     # created so that chucksize can be set in map
#     trainer, settings = train_tuple
#     return trainer.train(settings)
