from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import multiprocessing as mp

import utils
from mdp import common

if TYPE_CHECKING:
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder

_trainer: Trainer


class ParallelTrainer:
    def __init__(self, trainer: Trainer, parallel_context_type: common.ParallelContextType):
        self._trainer: Trainer = trainer
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        self._profile_child: bool = True
        self._parallel_context_type: common.ParallelContextType = parallel_context_type
        self._processes: int = os.cpu_count()

        self._settings_list: list[common.Settings] = []
        self._results: list[common.Result] = []
        self._recorder: Optional[Recorder] = None
        if self._trainer.breakdown:
            self._recorder = self._trainer.breakdown.recorder

        context_str: str = common.parallel_context_str[self._parallel_context_type]
        self._context: mp.context.BaseContext = mp.get_context(context_str)

    def train(self, settings_list: list[common.Settings]):
        self._settings_list = settings_list
        settings_count = len(self._settings_list)
        seeds: list[int] = utils.Rng.get_seeds(number_of_seeds=settings_count)
        profiles: list[bool] = [False for _ in range(settings_count)]
        if self._profile_child:
            profiles[0] = True

        # have final settings return everything (if used in case of V and Q)
        self._set_result_parameters()

        with self._context.Pool(processes=self._processes,
                                initializer=_init,
                                initargs=(self._trainer, )
                                ) as pool:
            args = zip(seeds,
                       profiles,
                       self._settings_list)
            self._results = pool.starmap(_train_wrapper, args)

        self._unpack_results()

        # set up agent using final setting and apply the final result
        self._trainer.apply_settings(settings=self._settings_list[-1])
        self._trainer.algorithm.apply_result(result=self._results[-1])

    def _set_result_parameters(self):
        for settings in self._settings_list[:-1]:
            settings.result_parameters = common.ResultParameters(
                return_recorder=True,
            )
        self._settings_list[-1].result_parameters = common.ResultParameters(
            return_recorder=True,
            return_policy_vector=True,
            return_v_vector=True,  # will return only if exists
            return_q_matrix=True   # will return only if exists
            )

    def _unpack_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown so is ready to be used for output
        if self._trainer.breakdown:
            unique_recorders = set(result.recorder for result in self._results)
            for recorder in unique_recorders:
                self._recorder.add_recorder(recorder)


def _init(trainer: Trainer):
    global _trainer
    _trainer = trainer


def _train_wrapper(seed: int,
                   profile: bool,
                   settings: common.Settings
                   ) -> common.Result:
    utils.Rng.set_seed(seed)
    if profile:
        import cProfile
        cProfile.runctx('_trainer.train(settings, return_result=True)',
                        globals(),
                        locals(),
                        'train_child.prof')
        print("train_child profiling")
    result: common.Result = _trainer.train(settings, return_result=True)
    return result
