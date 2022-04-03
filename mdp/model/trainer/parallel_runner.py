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

        self._settings = self._trainer.settings
        # settings.result_parameters.return_cum_timestep = True
        self._parallel_context_type: Optional[common.ParallelContextType] = self._settings.runs_multiprocessing
        self._runs = self._settings.runs

        self._results: list[common.Result] = []
        self._recorder: Optional[Recorder] = None
        if self._trainer.breakdown:
            self._recorder = self._trainer.breakdown.recorder

        # if self._parallel_context_type is None it should fail here
        context_str = common.parallel_context_str[self._parallel_context_type]
        self._ctx: mp.context.BaseContext = mp.get_context(context_str)
        self._use_global_trainer: bool = (self._parallel_context_type == common.ParallelContextType.FORK_GLOBAL)
        if self._use_global_trainer:
            global _trainer
            _trainer = self._trainer

    def do_runs(self):
        result_parameter_list: list[common.ResultParameters] = self._get_result_parameter_list()

        with self._ctx.Pool() as pool:
            if self._use_global_trainer:
                args = zip(range(1, self._runs + 1), result_parameter_list)
                # self._results = pool.map(_global_do_run_wrapper, args)
                self._results = pool.starmap(_global_do_run_wrapper, args)
            else:
                args = zip(itertools.repeat(self._trainer), range(1, self._runs + 1), result_parameter_list)
                # self._results = pool.map(_train_map_wrapper, args)
                self._results = pool.starmap(_do_run_starmap_wrapper, args)

        self._unpack_results()

        # the agent is already set up in trainer.trainer so just apply the final result to it
        self._trainer.algorithm.apply_result(result=self._results[-1])

    def _get_result_parameter_list(self) -> list[common.ResultParameters]:
        rp_norm: common.ResultParameters = common.ResultParameters(
            return_recorder=True,
            return_cum_timestep=True,
        )
        rp_final: common.ResultParameters = common.ResultParameters(
            return_recorder=True,
            return_cum_timestep=True,

            return_policy_vector=True,
            return_v_vector=True,
            return_q_matrix=True
        )

        result_parameter_list: list[common.ResultParameters] = list(itertools.repeat(rp_norm, self._runs-1))
        result_parameter_list.append(rp_final)
        return result_parameter_list

    def _unpack_results(self):
        # combine the recorders returned by the processes into a single recorder (self._recorder)
        # self._recorder is already attached to trainer via breakdown so is ready to be used for output
        if self._trainer.breakdown:
            unique_recorders = set(result.recorder for result in self._results)
            for recorder in unique_recorders:
                self._recorder.add_recorder(recorder)

        # self._final_settings.algorithm_title = self._results[-1].algorithm_title

        self._trainer.max_cum_timestep = max(result.cum_timestep for result in self._results)


def _global_do_run_wrapper(run_counter: int, result_parameters: common.ResultParameters)\
        -> common.Result:
    return _trainer.do_run(run_counter, result_parameters)


def _do_run_starmap_wrapper(trainer: Trainer, run_counter: int, result_parameters: common.ResultParameters)\
        -> common.Result:
    return trainer.do_run(run_counter, result_parameters)


# def _train_map_wrapper(train_tuple: tuple[Trainer, common.Settings]) -> common.Result:
#     # created so that chucksize can be set in map
#     trainer, settings = train_tuple
#     return trainer.train(settings)
