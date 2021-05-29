from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import multiprocessing as mp
import itertools

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder

# _trainer: Trainer


class ParallelTrainer:
    def __init__(self, trainer: Trainer):
        self._trainer: Trainer = trainer
        self._settings_list: list[common.Settings] = []
        self._ctx = mp.get_context('spawn')
        self._results: list[common.Result] = []
        self._recorder: Optional[Recorder] = self._trainer.breakdown.recorder

    def train(self, settings_list: list[common.Settings]):
        self._settings_list = settings_list
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        # have final settings return everything (if used in case of V and Q)
        self.alter_settings_to_return_everything(self._settings_list[-1])
        # global _trainer
        # _trainer = self._trainer

        with self._ctx.Pool() as pool:
            # self._results = pool.map(_global_train_wrapper, self._settings_list)
            self._results = pool.starmap(_train_wrapper, zip(itertools.repeat(self._trainer), self._settings_list))

        self._unpack_results()

        # set up agent using final setting and apply the final result
        self._trainer.agent.apply_result(settings=self._settings_list[-1], result=self._results[-1])

    def alter_settings_to_return_everything(self, settings: common.Settings):
        rp: common.ResultParameters = settings.result_parameters
        rp.return_policy_vector = True
        rp.return_v = True
        rp.return_q = True

    def _unpack_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown so is ready to be used for output
        for settings, result in zip(self._settings_list, self._results):
            settings.algorithm_title = result.algorithm_title
            self._recorder.add_recorder(result.recorder)

# def _global_train_wrapper(settings: common.Settings) -> Recorder:
#     return _trainer.train(settings)


def _train_wrapper(trainer: Trainer, settings: common.Settings) -> common.Result:
    # TODO: trainer is not always starting in a clean state. Why?
    print(f"len tallies: {len(trainer.breakdown.recorder.tallies)}")
    return trainer.train(settings)
