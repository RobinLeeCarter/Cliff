from __future__ import annotations
from typing import Optional
from multiprocessing import RLock
from multiprocessing.context import BaseContext
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager, BaseManager

import numpy as np

from utils.shared_array_door import SharedArrayDoor


class SharedArrayWrapper:
    def __init__(self):
        self._source: Optional[np.ndarray] = None
        self._shared_memory: Optional[SharedMemory] = None
        self._array: Optional[np.ndarray] = None
        self._door: Optional[SharedArrayDoor] = None
        self._lock: Optional[RLock] = None
        self._counter: int = 0

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def door(self) -> SharedArrayDoor:
        return self._door

    @property
    def lock(self) -> RLock:
        return self._lock

    def build(self,
              context: BaseContext,
              shared_memory_manager: BaseManager,
              source: np.ndarray) -> SharedArrayWrapper:
        # Pycharm or Python(?) type-checking fix
        assert isinstance(shared_memory_manager, SharedMemoryManager)

        self._source = source
        self._shared_memory = shared_memory_manager.SharedMemory(size=source.nbytes)
        self._array: np.ndarray = np.ndarray(shape=source.shape, dtype=source.dtype, buffer=self._shared_memory.buf)
        np.copyto(src=source, dst=self._array)

        self._door = SharedArrayDoor(
            lock=context.RLock(),
            name=self._shared_memory.name,
            shape=source.shape,
            dtype=source.dtype
        )
        self._lock = self._door.lock
        return self

    def attach(self, shared_array_door: SharedArrayDoor) -> SharedArrayWrapper:
        self._door = shared_array_door
        self._lock = self._door.lock
        self._shared_memory = SharedMemory(self._door.name)
        self._array: np.ndarray = np.ndarray(shape=self._door.shape,
                                             dtype=self._door.dtype,
                                             buffer=self._shared_memory.buf)
        return self

    def acquire(self):
        if self._lock:
            if self._counter == 0:
                self._lock.acquire()
            self._counter += 1

    def release(self):
        if self._lock:
            if self._counter == 1:
                self._lock.release()
            self._counter -= 1

    def copy_back(self):
        np.copyto(src=self._array, dst=self._source)

    def close(self):
        self._shared_memory.close()
        self._shared_memory = None
        self._door = None
        self._array = None
