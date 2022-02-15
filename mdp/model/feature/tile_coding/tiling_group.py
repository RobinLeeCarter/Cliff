from __future__ import annotations

import math
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from tile_code.dimension_range import DimensionRange


class TilingGroup:
    """
    Note: may actually need to "hash-down" to just the states that are actually used
    http://www.incompleteideas.net/RL-FAQ.html
    There are at least three common tricks:
    1) you can consider subsets of the state variables in separate tilings,
    2) you can use overlapping tilings to get vastly higher resolution in high dimensional spaces than would be
     possible with a simple grid using the same amount of memory, and
    3) you can hash down to a smaller space so as only to devote memory to the portions
     of state space that actually occur.
    """
    def __init__(self,
                 dimension_ranges: list[DimensionRange],
                 included_dims: np.ndarray,
                 tiles_per_dim: np.ndarray,
                 tilings: Optional[int] = None,
                 offset_per_dimension_fn: Optional[Callable[[int], np.ndarray]] = None):
        """
        :param dimension_ranges: value range of each dimension and whether it wraps
        :param included_dims: bool array of the dimensions to tile or not (based on dimension_ranges)
        :param tiles_per_dim: int tiles per dimension, determines size of generalisation, list all dimensions
        :param tilings: number of tilings e.g. n=8, should be a power of 2 and >= 4k (k = number of included dimensions)
        if unspecified or None, will use the lowest power of 2 above 4k
        :param offset_per_dimension_fn: fn that takes the number of dimensions and returns offset units per dimension
        """
        if offset_per_dimension_fn is None:
            offset_per_dimension_fn = self._first_odd_integers

        # restrict remaining code to just the dimensions the tiling is over
        # boolean arrays to select int and float dimensions being used
        self._included_ints = np.array([included and dimension_range.is_integer
                                        for dimension_range, included in zip(dimension_ranges, included_dims)])
        self._included_floats = np.array([included and not dimension_range.is_integer
                                          for dimension_range, included in zip(dimension_ranges, included_dims)])

        # float_dimensions = k in book
        float_dimensions: int = int(np.sum(self._included_floats))
        float_dimension_ranges: list[DimensionRange] = \
            [dimension_range for dimension_range, included in zip(dimension_ranges, self._included_floats) if included]
        # range minimum in each dimension
        self._float_dim_min = np.array([dimension_range.min for dimension_range in float_dimension_ranges])

        if tilings is None:
            if float_dimensions == 0:
                self._tilings = 1   # integer dimensions only
            else:
                self._tilings = 2 ** math.ceil(math.log2(4 * float_dimensions))
            # use number of tiles = number of tilings if unspecified
        else:
            self._tilings: int = tilings
            if not math.log2(self._tilings).is_integer():
                # TODO: Replace print statements with logging
                print("Warning: tilings not a power of 2")
            if self._tilings < 4 * float_dimensions:
                print("Warning: tilings < 4 * number of continuous tiling dimensions")

        # tiles needed per dimension ignoring offsets and wrapping for the moment
        # e.g. array([8, 8])
        if tilings is None or tiles_per_dim is None:
            float_requested_tiles_per_dim: list[int] = \
                [self._tilings for included in self._included_floats if included]
        else:
            float_requested_tiles_per_dim: list[int] = \
                [tiles for tiles, included in zip(tiles_per_dim, self._included_floats) if included]
        # calibrate input, dividing by the range and multiplying by the number of tiles. So a tile maps to 1.0.
        # e.g. array([8, 8]) / array([6.28318531, 6.28318531]) = array([1.27323954, 1.27323954])
        self._float_norm_tiles = np.array([
            float(tiles) / dimension_range.range
            for tiles, dimension_range in zip(float_requested_tiles_per_dim, float_dimension_ranges)
        ])

        # wrapped dimensions boolean array
        self._float_wrapped_dim = np.array([dr.wrap_around for dr in float_dimension_ranges], dtype=np.bool)
        # actual tiles per dimension (+ 1 extra tile needed for offset providing a full partitioning if not wrapped)
        # e.g. array([9, 9])
        float_actual_tiles_per_dim = np.array([
            tiles + int(not dr.wrap_around)
            for tiles, dr in zip(float_requested_tiles_per_dim, float_dimension_ranges)
        ], dtype=np.int)
        self._float_actual_tiles_per_wrapped_dim = float_actual_tiles_per_dim[self._float_wrapped_dim]

        # Calculate offsets

        offset_units_per_dimension = offset_per_dimension_fn(float_dimensions)
        # e.g. array([1, 3])

        tilings_array = np.arange(self._tilings)
        # e.g. array([0, 1, 2, 3, 4, 5, 6, 7])

        tilings_spread = np.repeat([tilings_array], float_dimensions, axis=0).T / float(self._tilings)
        """
        tilings spread evenly [0, 1) e.g. for 2 dimensions
        array([[0.   , 0.   ],
               [0.125, 0.125],
               [0.25 , 0.25 ],
               [0.375, 0.375],
               [0.5  , 0.5  ],
               [0.625, 0.625],
               [0.75 , 0.75 ],
               [0.875, 0.875]])
        """

        offset_lines = tilings_spread * offset_units_per_dimension  # element-wise multiplication
        """
        offset_lines is spread from [0, 1) and [0, 3) e.g.
        array([[0.   , 0.   ],
               [0.125, 0.375],
               [0.25 , 0.75 ],
               [0.375, 1.125],
               [0.5  , 1.5  ],
               [0.625, 1.875],
               [0.75 , 2.25 ],
               [0.875, 2.625]])              
        """

        self._float_offsets = offset_lines % 1
        """
        offset is mod 1 so spread 0 to 1 creating knight-move fill-in of space e.g.
        [[0.    0.   ]
         [0.125 0.375]
         [0.25  0.75 ]
         [0.375 0.125]
         [0.5   0.5  ]
         [0.625 0.875]
         [0.75  0.25 ]
         [0.875 0.625]]
        """

        int_ranges = np.array([
            math.floor(dimension_range.range + 1)   # + 1 because integer range with inclusive min and max
            for dimension_range, included
            in zip(dimension_ranges, self._included_ints) if included
        ], dtype=np.int)

        self._total_tiles = self._tilings * np.prod(float_actual_tiles_per_dim) * np.prod(int_ranges)

        # tiling_dims e.g. array([9, 9])
        # 81 * array([0, 1, 2, 3, 4, 5, 6, 7])
        # = array([  0,  81, 162, 243, 324, 405, 486, 567])
        # where the index for each tiling starts from e.g. first tile of 2nd tiling is index 81
        # self._tile_base_index = self._tile_coder_base_index + np.prod(self._tiles_per_dim) * np.arange(self._tilings)

        # e.g. self._tiles_per_dim = array([9, 9]) then array([1, 9])
        # docs: "The product of an empty array is the neutral element 1"
        # how much to increase index for off_coords in each axis (e.g. +1 for x-axis, +9 for y-axis)
        # potentially a bit odd as it means the rows are worth 1 and the columns are worth 9
        # might have expected the rows to be worth 9 and then columns to be worth 1 but both work
        # self._coords_flattening_vector = np.array([np.prod(self._tiles_per_dim[0:i]) for i in range(dimensions)])

        # total number of tiles e.g. 648
        # self._total_tiles: int = self._tilings * np.prod(self._tiles_per_dim)

    # perhaps should only return co-ords
    def __getitem__(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param state: full state values (floats and ints) as np.ndarray
        :return: tuple of array of float coords for each tiling + the int values played back for dimensions being tiled
        """
        float_tile_coords = np.floor(
            # state normalised so states from zero and tile width is 1.0
            (state[self._included_floats] - self._float_dim_min) * self._float_norm_tiles
            + self._float_offsets                               # broadcast to all tilings with their offsets
        ).astype(np.int)                                         # as floored ints
        """
        e.g. for (0.0, 0.0) 
        array([[0, 0],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 2],
               [0, 2]])
        """

        # wrap around if required
        float_tile_coords[:, self._float_wrapped_dim] %= self._float_actual_tiles_per_wrapped_dim

        int_values = state[self._included_ints].astype(np.int)

        return float_tile_coords, int_values

        # int_tile_coords = np.repeat(int_values[np.newaxis, :], 8, axis=0)
        #
        # # concatenate included ints, no need to deduce min
        # coords = np.concatenate((float_tile_coords, int_tile_coords), axis=1)
        # return coords

        # find tile index values from off_coords
        # return self._tile_base_index + np.dot(tile_coords, self._coords_flattening_vector)

    @staticmethod
    def _first_odd_integers(number_of_dimensions: int) -> np.ndarray:  # e.g. 2 -> array([1, 3])
        return 2 * np.arange(number_of_dimensions) + 1

    @property
    def total_tiles(self) -> int:
        return self._total_tiles

    @property
    def tilings(self) -> int:
        return self._tilings
