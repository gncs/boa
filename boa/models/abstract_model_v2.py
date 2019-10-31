import abc

import numpy as np
import tensorflow as tf


class ModelError(Exception):
    """Base error thrown by models"""


class AbstractModel(tf.keras.Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str = "abstract_model", **kwargs):

        super(AbstractModel, self).__init__(name=name, **kwargs)

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