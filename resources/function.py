"""
Example for objective function that can be optimized with the BOA program
"""
from typing import List

import numpy as np

from boa.objective.abstract import AbstractObjective


class Objective(AbstractObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input_labels(self) -> List[str]:
        """Return input labels as a list of length D_input"""
        return ["x0", "x1", ]

    def get_output_labels(self) -> List[str]:
        """Return output labels as a list of length D_output"""
        return ["y0", "y1", ]

    def get_candidates(self) -> np.ndarray:
        """Return potential candidates as an array of shape N x D_input"""
        return np.array([np.array([i, i + 1, ]) for i in np.arange(0, 1.5, 0.01)])

    def __call__(self, value: np.ndarray) -> np.ndarray:
        """Return output of objective function as an array of shape N x D_output"""
        return np.array([np.sin(5 * value[0]), np.cos(3 * value[1])])
