import abc
from multiprocessing.pool import ThreadPool
from typing import Tuple, List

import numpy as np


class AbstractObjective:
    __metaclass__ = abc.ABCMeta

    def __init__(self, pool_size: int = 1, *args, **kwargs):
        self.pool_size = pool_size

    @abc.abstractmethod
    def get_candidates(self) -> np.ndarray:
        """Return potential candidates as an array of shape N x D_input"""

    @abc.abstractmethod
    def get_input_labels(self) -> List[str]:
        """Return input labels as a list of length D_input"""

    @abc.abstractmethod
    def get_output_labels(self) -> List[str]:
        """Return output labels as a list of length D_output"""

    @abc.abstractmethod
    def __call__(self, candidate: np.ndarray) -> np.ndarray:
        """
        Evaluate candidate (1D array of length D_input) and return value as 1D array of length D_output.
        Note: in parallel mode, this function is called by multiple processes.
        Therefore, this function must not alter the state of this object (self)!
        """

    def evaluate_batch(self, candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate objective for list of parameters <candidates>.

        :param candidates: array of dimension N x D_input
        :return: list of (input, output) pairs
        """

        with ThreadPool(self.pool_size) as p:
            results = p.map(self._evaluate, candidates)

        xs = []
        ys = []

        for candidate, result in results:
            xs.append(candidate)
            ys.append(result)

        return np.vstack(xs), np.vstack(ys)

    def _evaluate(self, candidate):
        return candidate, self(candidate)
