from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import multiprocessing as mp
import itertools

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder


class ParallelTrainer:
    def __init__(self, trainer: Trainer):
        self._trainer: Trainer = trainer
        self._settings_list: list[common.Settings] = []
        self._ctx = mp.get_context('spawn')
        self._results: list[Recorder] = []
        self._recorder: Optional[Recorder] = self._trainer.breakdown.recorder

    def train(self, settings_list: list[common.Settings]):
        self._settings_list = settings_list
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        with self._ctx.Pool() as pool:
            self._results = pool.starmap(_train_wrapper, zip(itertools.repeat(self._trainer), self._settings_list))
        self._compile_results()

    def _compile_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown
        for recorder in self._results:
            for key, tally in recorder.tallies.items():
                self._recorder.add_tally(key, tally)


def _train_wrapper(trainer: Trainer, settings: common.Settings) -> Recorder:
    return trainer.train(settings)
