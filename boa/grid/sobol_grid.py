from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pyscenarios.sobol import sobol

from boa.core import InputSpec

from .grid import Grid

tfd = tfp.distributions


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

        self.points = self._from_cube_to_grid_points(self.cube_grid)

        self.points = tf.convert_to_tensor(self.points)
        self.points = tf.cast(self.points, tf.float64)

    def _from_cube_to_grid_points(self, cube_points):

        points = np.empty_like(cube_points)

        # Convert these points to valid settings
        for dim, spec in enumerate(self.dim_spec):
            # Can't just scale, shift then round -> there might be gaps in domain
            # Hence, sample the index first, then select item from domain with index
            indices = cube_points[:, dim] * len(spec.domain)
            indices = np.floor(indices).astype(np.int32)

            points[:, dim] = spec.domain[indices]

        # Eliminate duplicate grid points
        points = np.unique(points, axis=0)

        return points

    def sample(self, num_grid_points, seed=None):
        if num_grid_points > self.max_grid_points:
            raise ValueError(f"num_grid_points ({num_grid_points}) cannot exceed the maximum number "
                             f"of grid points ({self.max_grid_points})!")

        samples = tf.random.shuffle(self.points, seed=seed)[:num_grid_points, :]

        return samples

    def sample_grid_around_point(self, point, num_samples, scale=0.3, seed=None, add_to_grid=False):
        # Note: a scale (std dev) of 0.3 corresponds to approximately a 90% of not changing for a single dimension
        if len(point.shape) != 1:
            raise ValueError(f"Passed point must be rank-1, but had shape: {point.shape}")

        # Build bounds on what the indices can be
        high_bounds = np.empty_like(point)
        low_bounds = np.empty_like(point)

        for dim, spec in enumerate(self.dim_spec):
            # How many items in the domain are larger than the current points dimension
            high_bounds[dim] = np.sum(spec.domain > point[dim]).astype(high_bounds.dtype)

            # How many items in the domain are smaller than the current points dimension. Note the negative
            low_bounds[dim] = -np.sum(spec.domain < point[dim]).astype(low_bounds.dtype)

        # The location is set to -0.5, because the distribution is quantized on the intervals
        # ... (-2, -1] -> -1, (-1, 0] -> 0, (0, 1] -> 1 ...
        index_change_distribution = tfd.QuantizedDistribution(distribution=tfd.Normal(loc=-0.5 * tf.ones_like(point),
                                                                                      scale=scale),
                                                              high=high_bounds,
                                                              low=low_bounds)

        index_changes = index_change_distribution.sample(num_samples).numpy()

        # Eliminate identical changes
        index_changes = np.unique(index_changes, axis=0)
        index_changes = tf.convert_to_tensor(index_changes, point.dtype)

        samples = point[None, :] + index_changes

        if add_to_grid:
            new_points = tf.concat([self.points, samples], axis=0).numpy()
            new_points = np.unique(new_points, axis=0)
            self.points = tf.convert_to_tensor(new_points, dtype=tf.float64)

        return samples