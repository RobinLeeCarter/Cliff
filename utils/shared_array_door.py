from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import Lock


@dataclass(frozen=True)
class SharedArrayDoor:
    lock: Lock
    name: str
    shape: tuple[int]
    dtype: type
