from __future__ import annotations
# from typing import TYPE_CHECKING

from mdp.common.dataclass.result import Result
from mdp.model.breakdown.recorder import Recorder


recorder = Recorder[int]()
results = Result(recorder=recorder)
