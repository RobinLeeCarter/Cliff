from __future__ import annotations
from dataclasses import dataclass

from mdp.model.environment import response
from mdp.scenarios.jacks.state import State


@dataclass(frozen=True)
class Response(response.Response):
    state: State
