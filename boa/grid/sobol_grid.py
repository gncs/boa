from typing import List

import numpy as np
import tensorflow as tf

import networkx as nx

from pyscenarios.sobol import sobol

from boa.core import InputSpec

from .grid import Grid


class SobolGrid(Grid):
    """
    TODO: Handle categorical variables.

    TODO: Spearmint adds additional candidates around current best solution:
    https://github.com/HIPS/Spearmint/blob/990d27d4477bbc9b0d5cfb2c950fc387decf6ea2/spearmint/choosers/default_chooser.py#L338
    """

    def __init__(self,
                 dim_specs: List[InputSpec],
                 seed: int,
                 fraction: float = 0.,
                 num_grid_points: int = 0,
                 skip: int = 2000,
                 **kwargs):

        super().__init__(dim_specs, **kwargs)

        # If num_grid_points is given
        if fraction <= 0. and num_grid_points > 0:

            if self.max_grid_points < num_grid_points:
                raise ValueError(f"num_grid_points ({num_grid_points}) must be less than "
                                 f"the maximum possible grid points ({self.max_grid_points})!")

            self.num_grid_points = num_grid_points

        elif fraction > 0. and num_grid_points <= 0:

            if fraction >= 1.:
                raise ValueError(f"percent must bel less than 1, but {fraction} was given.")

            self.num_grid_points = int(np.ceil(fraction * self.max_grid_points))

        else:
            raise ValueError(f"Either percent ({fraction}) or num_grid_points ({num_grid_points}) "
                             f"has to be valid!")

        # Create grid
        print(f"Creating {self.__class__.__name__} with {self.num_grid_points} points!")

        # sample points on the unit hypercube
        # Note: this sobol function can only handle input vectors up to dimension 21201.
        self.cube_grid = sobol(size=(skip + self.num_grid_points, self.dimension), d0=seed)[skip:, :]
        self.points = np.empty_like(self.cube_grid)

        # Convert these points to valid settings
        for dim, spec in enumerate(self.dim_spec):
            # Can't just scale, shift then round -> there might be gaps in domain
            # Hence, sample the index first, then select item from domain with index
            indices = self.cube_grid[:, dim] * len(spec.domain)
            indices = np.floor(indices).astype(np.int32)

            self.points[:, dim] = spec.domain[indices]

        # Eliminate duplicate grid points
        self.points = np.unique(self.points, axis=0)

        self.points = tf.convert_to_tensor(self.points)
        self.points = tf.cast(self.points, tf.float64)

    def sample(self, num_grid_points, seed=None):
        if num_grid_points > self.max_grid_points:
            raise ValueError(f"num_grid_points ({num_grid_points}) cannot exceed the maximum number "
                             f"of grid points ({self.max_grid_points})!")

        samples = tf.random.shuffle(self.points, seed=seed)[:num_grid_points, :]

        return samples


