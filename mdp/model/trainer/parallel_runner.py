from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import multiprocessing as mp
import itertools

from mdp import common

if TYPE_CHECKING:
    from mdp.model.trainer.trainer import Trainer
    from mdp.model.breakdown.recorder import Recorder

_trainer: Trainer


class ParallelRunner:
    def __init__(self, trainer: Trainer):
        self._trainer: Trainer = trainer
        # must be disabled for multi-processor
        self._trainer.disable_step_callback()

        settings = self._trainer.settings
        settings.result_parameters.return_cum_timestep = True
        self._parallel_context_type = settings.runs_multiprocessing
        self._runs = settings.runs
        # build a settings list for each run so that the final run can return everything
        self._settings_list: list[common.Settings] = list(itertools.repeat(settings, self._runs))

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

    def do_runs(self):
        # have final settings return everything (if used in case of V and Q)
        self.alter_settings_to_return_everything(self._settings_list[-1])

        with self._ctx.Pool() as pool:
            if self._use_global_trainer:
                self._results = pool.map(_global_train_wrapper, self._settings_list)
            else:
                args = zip(itertools.repeat(self._trainer), self._settings_list, range(1, self._runs + 1))
                # self._results = pool.map(_train_map_wrapper, args)
                self._results = pool.starmap(_train_starmap_wrapper, args)

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


def _global_train_wrapper(settings: common.Settings, run_counter: int) -> common.Result:
    return _trainer.do_run(settings, run_counter)


def _train_starmap_wrapper(trainer: Trainer, settings: common.Settings, run_counter: int) -> common.Result:
    return trainer.do_run(settings, run_counter)


# def _train_map_wrapper(train_tuple: tuple[Trainer, common.Settings]) -> common.Result:
#     # created so that chucksize can be set in map
#     trainer, settings = train_tuple
#     return trainer.train(settings)
