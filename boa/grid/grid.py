import abc
from typing import List
import networkx

import numpy as np

from boa.core import InputSpec


class Grid(abc.ABC):

    def __init__(self, dim_spec: List[InputSpec], verbose=True):

        if not isinstance(dim_spec, List) or \
                not all(map(lambda x: isinstance(x, InputSpec), dim_spec)):
            raise TypeError("dim_spec must be a list of InputSpecs!")

        # Convert all domains to numpy arrays
        self.dim_spec = [InputSpec(name, np.array(domain).astype(np.float64))
                         for name, domain in dim_spec]
        self._max_grid_points = 1

        for _, domain in dim_spec:
            self._max_grid_points *= len(domain)

        if verbose:
            print(f"Maximum number of grid points: {self.max_grid_points}")

    @property
    def max_grid_points(self) -> int:
        return self._max_grid_points

    @property
    def dimension(self) -> int:
        return len(self.dim_spec)

    def sample(self, num_grid_points, seed=None):
        pass
