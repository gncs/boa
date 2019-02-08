import abc

import numpy as np

from boa.models.abstract import AbstractModel


class AbstractAcquisition:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, model: AbstractModel, xs: np.ndarray, ys: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        pass
