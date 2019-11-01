import abc

import numpy as np

from boa.models.abstract_model_v2 import AbstractModel


class AbstractAcquisition:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, model: AbstractModel, xs: np.ndarray, ys: np.ndarray, candidate_xs: np.ndarray) -> np.ndarray:
        pass
