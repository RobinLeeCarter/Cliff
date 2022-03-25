from __future__ import annotations

from mdp import common
from mdp.task.cliff.comparison.comparison_builder import ComparisonBuilder
from mdp.task.cliff.comparison.comparison import Comparison
from mdp.task.cliff.model.environment_parameters import EnvironmentParameters


class CliffAlphaEnd(ComparisonBuilder):
    def create(self) -> Comparison:
        # TODO: Make work, once multiprocessing
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=common.Settings(
                runs=1,
                training_episodes=100_000,
            ),
            breakdown_parameters=common.BreakdownAlgorithmByAlpha(
                breakdown_type=common.BreakdownType.RETURN_BY_ALPHA,
                alpha_min=0.1,
                alpha_max=1.0,
                alpha_step=0.05,
                algorithm_type_list=[
                    common.AlgorithmType.TABULAR_EXPECTED_SARSA,
                    # common.AlgorithmType.VQ,
                    common.AlgorithmType.TABULAR_Q_LEARNING,
                    common.AlgorithmType.TABULAR_SARSA
                ],
            ),
            settings_list_multiprocessing=common.ParallelContextType.SPAWN,
            graph2d_values=common.Graph2DValues(
                has_grid=True,
                has_legend=True,
                y_min=-140,
                y_max=0,
            ),
        )
