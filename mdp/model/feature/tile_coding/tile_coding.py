from __future__ import annotations

from typing import Optional, Callable, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.dimension_enum import DimensionEnum
    from mdp.model.environment.non_tabular.dimension.dimension import Dimension
    from mdp.model.environment.non_tabular.dimension.float_dimension import FloatDimension
    from mdp.model.environment.non_tabular.dimension.category_dimension import CategoryDimension
from mdp.model.feature.tile_coding.tiling_group import TilingGroup


class TileCoding:
    def __init__(self,
                 state_float_dimensions: dict[DimensionEnum, FloatDimension],
                 state_category_dimensions: dict[DimensionEnum, CategoryDimension],
                 action_category_dimensions: dict[DimensionEnum, CategoryDimension],
                 max_size: Optional[int] = None,
                 use_dict: bool = True):
        """
        :param dimension_ranges: value range of each dimension and whether it wraps
        :param max_size: optional: maximum tile number, will reuse from start if using a dict and show a warning
        :param use_dict: pass False is wish to Hash % max_size and avoid using a dict
        """
        self._state_float_dimensions: dict[DimensionEnum, FloatDimension] = state_float_dimensions
        self._state_category_dimensions: dict[DimensionEnum, CategoryDimension] = state_category_dimensions
        self._action_category_dimensions: dict[DimensionEnum, CategoryDimension] = action_category_dimensions
        self._max_size: Optional[int] = max_size
        self._use_dict: bool = use_dict

        self._tiling_groups: list[TilingGroup] = []     # OK initialisation ???
        self._total_tilings: int = 0
        self._total_tiles: int = 0
        self._exceeded_max_size: bool = False

        # dictionary of coord to tile_indexes
        self._tile_dict: dict[tuple[int, ...], int] = {}
        # counter of tile_indexes for dictionary
        self._tile_index_counter: int = 0

    def add(self,
            included_dimensions: list[DimensionEnum],
            included_dims: np.ndarray,
            tile_size_per_dim: Optional[np.ndarray] = None,
            tiles_per_dim: Optional[np.ndarray] = None,
            tilings: Optional[int] = None,
            offset_per_dimension_fn: Optional[Callable[[int], np.ndarray]] = None
            ):
        """
        :param included_dimensions: unordered list of dimensions to include in the tile coding group
        :param included_dims: bool array of the dimensions to tile or not (based on dimension_ranges)
        :param tile_size_per_dim: either specify tile_size in each dimension (distance over which to generalise)...
        :param tiles_per_dim: or specify numbers of tiles in each dimension (all dimensions not just included ones)
        :param tilings: number of tilings e.g. n=8, should be a power of 2 and >= 4k (k = number of included dimensions)
        if unspecified or None, will use the lowest power of 2 above 4k
        :param offset_per_dimension_fn: fn that returns offset units in each dimension given the number of dimensions
        """

        # convert to boolean arrays, note that dicts are safely ordered as of python 3.7
        # TODO: is this best done in tiling_group
        included_state_floats: np.ndarray = np.array(
            [dimension in included_dimensions for dimension in self._state_float_dimensions.keys()])
        included_state_category: np.ndarray = np.array(
            [dimension in included_dimensions for dimension in self._state_category_dimensions.keys()])
        included_action_category: np.ndarray = np.array(
            [dimension in included_dimensions for dimension in self._action_category_dimensions.keys()])

        if tile_size_per_dim is not None and tiles_per_dim is not None:
            print("Warning: specify tile_size_per_dim or tiles_per_dim, not both.")
        # if tile_size_per_dim is None and tiles_per_dim is None:
        #     print("Warning: specify either tile_size_per_dim or tiles_per_dim.")
        if tiles_per_dim is None and tile_size_per_dim is not None:
            tiles_per_dim = np.array(
                [dimension_range.calc_tiles(tile_size)
                 for dimension_range, tile_size in zip(self._dimension_ranges, tile_size_per_dim)]
            ).astype(np.int)
        tiling_group = TilingGroup(self._dimension_ranges,
                                   included_dims,
                                   tiles_per_dim,
                                   tilings,
                                   offset_per_dimension_fn)
        self._tiling_groups.append(tiling_group)
        self._total_tilings += tiling_group.tilings
        if self._use_dict:
            self._total_tiles += tiling_group.total_tiles
            # if self._size != 0 and self._total_tiles > self._size:
            #     print(f"Warning: total_tiles: {self._total_tiles} exceeds size: {self._size}")

    def __getitem__(self, state_values: tuple[float, ...]) -> np.ndarray:
        """
        :param state_values: full state values as np.ndarray
        :return: np.ndarray of active tile indexes
        """
        state = np.array(state_values)
        tile_results_list: list[tuple[np.ndarray, np.ndarray]]\
            = [tile_coder[state] for tile_coder in self._tiling_groups]
        tile_indexes: list[int] = []
        for tiling_group_index, (tile_coords, int_values) in enumerate(tile_results_list):
            int_tuple = tuple(int_values)
            for tiling, coord in enumerate(tile_coords):
                full_coord_tuple: tuple[int, ...] = (tiling_group_index, tiling) + tuple(coord) + int_tuple
                tile_index: int = self._get_tile_index(full_coord_tuple)
                tile_indexes.append(tile_index)
        return np.array(tile_indexes, dtype=np.int)

    def _get_tile_index(self, full_coord_tuple: tuple[int, ...]) -> int:
        """
        :param full_coord_tuple: full coord includes tiling_group, tiling, float coords and int coords
        :return: the tile index for the full_coord_tuple
        """
        if self._use_dict:
            # want to get value if present, else set value and increment
            tile_index = self._tile_dict.get(full_coord_tuple)   # avoids raising KeyError
            if tile_index is None:
                if self._exceeded_max_size:
                    tile_index = hash(full_coord_tuple) % self._max_size
                else:
                    tile_index = self._tile_index_counter
                    self._tile_dict[full_coord_tuple] = tile_index
                    self._tile_index_counter += 1
                    if self._max_size is not None and self._tile_index_counter >= self._max_size:
                        # self._tile_index_counter %= self._max_size
                        self._exceeded_max_size = True
                        print(f"Warning: Collisions, tile dict indexes to exceed output size of {self._max_size}")
        else:
            tile_index = hash(full_coord_tuple) % self._max_size
        return tile_index

    @property
    def count(self) -> int:
        """
        could be that few tiles and therefore weights are needed as only in a subset of the space is being used
        :return: "The number of indices currently used.
        All used indices will be between 0 and one less than this number.
        The count is always less than the size, and only count have been used.
        Knowing the count can often speed operations that only need be done on the used features
        (e.g., eligibility traces)."
        """
        if self._use_dict:
            if self._exceeded_max_size:
                return self._max_size
            else:
                return self._tile_index_counter + 1
        else:
            return self._max_size

    @property
    def max_size(self) -> int:
        """
        :return: maximum number of possible tiles can can be returned ever from this tile coding
        """
        if self._use_dict:
            if self._max_size is not None:
                return min(self._total_tiles, self._max_size)
            else:
                return self._total_tiles
        else:
            return self._max_size

    @property
    def tilings(self) -> int:
        """
        :return: total number of tilings from all tiling groups (used to set step-size alpha)
        """
        return self._total_tilings
