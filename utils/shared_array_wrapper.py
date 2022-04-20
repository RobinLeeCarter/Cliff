from __future__ import annotations
from typing import Optional
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from multiprocessing_shared_memory.shared_array_door import SharedArrayDoor


class SharedArrayWrapper:
    def __init__(self, *,
                 shared_memory_manager: Optional[SharedMemoryManager] = None,
                 source: Optional[np.ndarray] = None,
                 shared_array_door: Optional[SharedArrayDoor] = None):
        self._shared_memory_manager: Optional[SharedMemoryManager] = shared_memory_manager
        self._source: Optional[np.ndarray] = source
        self._door: Optional[SharedArrayDoor] = shared_array_door

        self._shared_memory: Optional[SharedMemory] = None
        self._array: Optional[np.ndarray] = None
        # self._lock: Optional[Lock] = None

        if shared_memory_manager is not None and source is not None:
            # parent process
            self._copy_in(source)
        elif shared_array_door is not None:
            # child process
            self._attach()

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def door(self) -> SharedArrayDoor:
        return self._door

    @property
    def lock(self) -> Lock:
        return self._door.lock

    def _copy_in(self, source: np.ndarray):
        self._shared_memory = self._shared_memory_manager.SharedMemory(size=source.nbytes)
        self._array: np.ndarray = np.ndarray(shape=source.shape, dtype=source.dtype, buffer=self._shared_memory.buf)
        # shared_array_manager=self)
        np.copyto(src=source, dst=self._array)
        # self._lock = Lock()
        self._door = SharedArrayDoor(
            lock=Lock(),
            name=self._shared_memory.name,
            shape=source.shape,
            dtype=source.dtype
        )

    def _attach(self):
        self._shared_memory = SharedMemory(self._door.name)
        self._array: np.ndarray = np.ndarray(shape=self._door.shape,
                                             dtype=self._door.dtype,
                                             buffer=self._shared_memory.buf)
        # shared_array_manager=self)
        # self._lock: Lock = door.lock

    # def acquire(self):
    #     self._lock.acquire()
    #
    # # possible usage?
    # def fetch(self) -> np.ndarray:
    #     self._lock.acquire()
    #     return self._array
    #
    # def release(self):
    #     self._lock.release()

    def copy_back(self):
        np.copyto(src=self._array, dst=self._source)

    def close(self):
        self._shared_memory.close()
        self._shared_memory = None
        self._door = None
        self._array = None
