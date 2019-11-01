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
    def __or__(self, inputs) -> tf.keras.Model:
        if not isinstance(inputs, tuple) or len(inputs) != 2:
            raise ModelError("Input must be a tuple of (xs, ys)!")

        xs, ys = inputs

        # Reasonable test of whether the inputs are array-like
        if not hasattr(xs, "__len__") or not hasattr(ys, "__len__"):
            raise ModelError("xs and ys must be array-like!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)
        ys = tf.convert_to_tensor(ys, dtype=tf.float64)

        # Check if the shapes are correct
        if not len(xs.shape) == 2 or not len(ys.shape) == 2:
            raise ModelError("xs and ys must be of rank 2!")

        # Ensure the user provided the same number of input and output points
        if not xs.shape[0] == ys.shape[0]:
            raise ModelError("The first dimension of xs and ys must be equal!")

        return None

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