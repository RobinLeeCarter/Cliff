from __future__ import annotations

from mdp import common, controller
from mdp.model import model_
from mdp.view import view_


class Application:
    def __init__(self, comparison: common.Comparison):
        self._comparison: common.Comparison = comparison

        self._model: model_.Model = model_.Model()
        self._view: view_.View = view_.View()
        self._controller: controller.Controller = controller.Controller(self._model, self._view)

        self.build()
        self.run()

    def build(self):
        # enable model and view to send messages to controller
        self._model.set_controller(self._controller)
        self._view.set_controller(self._controller)

        self._controller.build(self._comparison)

    def run(self):
        self._controller.run()
