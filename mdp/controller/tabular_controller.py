from __future__ import annotations
from typing import Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
from mdp.controller.base_controller import BaseController
from mdp.model.tabular.tabular_model import TabularModel
from mdp.view.tabular.tabular_view import TabularView

Model = TypeVar("Model", bound=TabularModel)
View = TypeVar("View", bound=TabularView)


class TabularController(BaseController[Model, View]):
    # region Model requests
    def display_step(self, episode: Optional[TabularEpisode]):
        # if self._comparison.grid_view_parameters.show_step:
        self._view.grid_view.display_latest_step(episode)
    # endregion

    # region View requests
    def new_episode_request(self) -> TabularEpisode:
        return self._model.agent.generate_episode()
    # endregion
