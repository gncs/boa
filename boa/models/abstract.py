import abc

import numpy as np


class AbstractModel:
    __metaclass__ = abc.ABCMeta

    def __index__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def set_data(self, xs: np.ndarray, ys: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def add_true_point(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def add_pseudo_point(self, x: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def remove_pseudo_points(self) -> None:
        pass

    @abc.abstractmethod
    def train(self) -> None:
        pass

    @abc.abstractmethod
    def predict_batch(self, xs: np.ndarray) -> np.ndarray:
        pass
