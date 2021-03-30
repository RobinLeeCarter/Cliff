from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model.agent.episode import Episode
    from mdp.model.model import Model
    from mdp.view.view import View
    from mdp import common


class Controller:
    def __init__(self):
        self._model: Optional[Model] = None
        self._view: Optional[View] = None
        self._comparison: Optional[common.Comparison] = None

    def link_mvc(self, model: Model, view: View):
        self._model: Model = model
        self._view: View = view
        self._model.set_controller(self)
        self._view.set_controller(self)

    def build(self, comparison: common.Comparison):
        # self._view.open() to determine user environment only
        self._comparison = comparison
        self._model.build(self._comparison)
        # TODO: don't pass grid_world here - too specific
        self._view.build(grid_world_=self._model.environment.grid_world, comparison=self._comparison)

    def run(self):
        self._model.run()

    def output(self):
        if self._comparison.graph_values.show_graph:
            # TODO: Overhaul
            # self._model.environment.insert_state_function_into_graph2d(
            #     self._comparison,
            #     self._model.agent.algorithm.V
            # )
            # self._view.graph.make_plot(self._comparison.graph_values)
            #
            # self._model.environment.insert_policy_into_graph2d(
            #     self._comparison,
            #     self._model.agent.policy
            # )
            # self._view.graph.make_plot(self._comparison.graph_values)

            if self._model.breakdown:
                graph_values: common.GraphValues = self._model.breakdown.get_graph_values()
                self._view.graph.make_plot(graph_values)

        g3d = self._comparison.graph3d_values
        if g3d.show_graph:
            if g3d.multi_parameter:
                for parameter in g3d.multi_parameter:
                    self._model.environment.insert_state_function_into_graph3d(
                        self._comparison,
                        self._model.agent.algorithm.V,
                        parameter
                    )
                    self._view.graph3d.make_plot(self._comparison.graph3d_values)
            else:
                self._model.environment.insert_state_function_into_graph3d(
                    self._comparison,
                    self._model.agent.algorithm.V
                )
                self._view.graph3d.make_plot(self._comparison.graph3d_values)

        gvp = self._comparison.grid_view_parameters
        if gvp.show_result or gvp.show_demo:
            if g3d.multi_parameter and gvp.show_result:
                for parameter in g3d.multi_parameter:
                    self._model.prep_for_output(parameter)
                    self._view.grid_view.display_parameter(parameter)
                    self._view.grid_view.display_latest_step()
            else:
                self._model.prep_for_output()
                if gvp.show_result:
                    self._view.grid_view.display_latest_step()
                if gvp.show_demo:
                    self._view.grid_view.demonstrate(self.new_episode_request)

    # region Model requests
    # def display_graph_2d(self, graph_values: common.GraphValues):
    #     self._view.graph.make_plot(graph_values)

    def display_step(self, episode_: Optional[Episode]):
        # if self._comparison.grid_view_parameters.show_step:
        self._view.grid_view.display_latest_step(episode_)
    # endregion

    # region View requests
    def new_episode_request(self) -> Episode:
        return self._model.agent.generate_episode()
    # endregion
