from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import RLock


@dataclass(frozen=True)
class SharedArrayDoor:
    lock: RLock
    name: str
    shape: tuple[int]
    dtype: type
