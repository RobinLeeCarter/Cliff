from __future__ import annotations

import environment


class Actions(environment.Actions):
    def _build_action_list(self):
        self._four_actions()
        # self._four_friendly_actions()
        # self._kings_moves()
        # self._kings_moves(include_center=True)
