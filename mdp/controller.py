from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import agent

from mdp import model, view
import common


class Controller:
    def __init__(self, model_: model.Model, view_: view.View):
        self._model: model.Model = model_
        self._view: view.View = view_

    def build(self, comparison: common.Comparison):
        # self._view.open() to determine user environment only
        self._model.build(comparison)
        self._view.build(grid_world_=self._model.environment.grid_world)

    # view requests
    def get_fresh_episode(self) -> agent.Episode:
        return self._model.agent.get_fresh_episode()
