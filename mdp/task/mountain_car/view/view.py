from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.task.mountain_car.controller import Controller

from mdp.view.non_tabular.non_tabular_view import NonTabularView


class View(NonTabularView):
    def __init__(self):
        super().__init__()
        self._controller: Optional[Controller] = None
