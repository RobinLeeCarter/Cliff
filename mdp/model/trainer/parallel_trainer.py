from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import multiprocessing as mp
import itertools

from mdp import common

if TYPE_CHECKING:
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder

_trainer: Trainer

# TODO: decide how to parameterise the different context selections including 'fork_without_pickle'
#  And this is probably the method you would want to use most for 'interesting'/large problems


class ParallelTrainer:
    def __init__(self, trainer: Trainer):
        self._trainer: Trainer = trainer
        self._settings_list: list[common.Settings] = []
        self._ctx = mp.get_context('fork')
        self._results: list[common.Result] = []
        self._recorder: Optional[Recorder] = None
        if self._trainer.breakdown:
            self._recorder = self._trainer.breakdown.recorder

    def train(self, settings_list: list[common.Settings]):
        self._settings_list = settings_list
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        # have final settings return everything (if used in case of V and Q)
        self.alter_settings_to_return_everything(self._settings_list[-1])

        global _trainer
        _trainer = self._trainer

        # args = zip(itertools.repeat(self._trainer), self._settings_list)
        with self._ctx.Pool() as pool:
            self._results = pool.map(_global_train_wrapper, self._settings_list)
            # self._results = pool.starmap(_train_starmap_wrapper, args)
            # self._results = pool.map(_train_map_wrapper, args)

        self._unpack_results()

        # set up agent using final setting and apply the final result
        self._trainer.agent.apply_result(settings=self._settings_list[-1], result=self._results[-1])

    def alter_settings_to_return_everything(self, settings: common.Settings):
        rp: common.ResultParameters = settings.result_parameters
        rp.return_policy_vector = True
        rp.return_v_vector = True
        rp.return_q_matrix = True

    def _unpack_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown so is ready to be used for output
        if self._trainer.breakdown:
            unique_recorders = set(result.recorder for result in self._results)
            for recorder in unique_recorders:
                self._recorder.add_recorder(recorder)

        for settings, result in zip(self._settings_list, self._results):
            settings.algorithm_title = result.algorithm_title


def _global_train_wrapper(settings: common.Settings) -> common.Result:
    return _trainer.train(settings)


def _train_starmap_wrapper(trainer: Trainer, settings: common.Settings) -> common.Result:
    return trainer.train(settings)


def _train_map_wrapper(train_tuple: tuple[Trainer, common.Settings]) -> common.Result:
    # created so that chucksize can be set in map
    trainer, settings = train_tuple
    return trainer.train(settings)
