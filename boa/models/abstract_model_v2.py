from typing import List

import abc

import numpy as np
import tensorflow as tf

from stheno.tensorflow import GP, EQ, Delta, Matern52, Graph


class ModelError(Exception):
    """Base error thrown by models"""


class AbstractModel(tf.keras.Model):

    AVAILABLE_KERNELS = {"rbf": EQ,
                         "matern52": Matern52, }

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 parallel: bool = False,
                 verbose: bool = False,
                 init_minval: float = 0.5,
                 init_maxval: float = 2.0,
                 name: str = "abstract_model", **kwargs):

        super(AbstractModel, self).__init__(name=name, **kwargs)

        # Independent GPs for each output dimension
        self.models: List = []

        # Check if the specified kernel is available
        if kernel in self.AVAILABLE_KERNELS:
            self.kernel_name = kernel
        else:
            raise ModelError("Specified kernel {} not available!".format(kernel))

        self.num_optimizer_restarts = num_optimizer_restarts
        self.parallel = parallel

        self.input_dim = 0
        self.output_dim = 0

        self.verbose = verbose

        self.xs_mean = tf.constant([[]])
        self.xs_std = tf.constant([[]])

        self.ys_mean = tf.constant([[]])
        self.ys_std = tf.constant([[]])

        self.xs = tf.constant([[]])
        self.ys = tf.constant([[]])

        self.num_pseudo_points = 0
        self.num_true_points = 0

        # Range for the initialisation of GP hyperparameters
        self.init_minval = tf.constant(init_minval, dtype=tf.float64)
        self.init_maxval = tf.constant(init_maxval, dtype=tf.float64)

    @abc.abstractmethod
    def __or__(self, inputs) -> tf.keras.Model:
        """
        Adds data to the model. The notation is supposed to imitate
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param inputs: Tuple of two rank-2 tensors: the first
        N x I and the second N x O, where N is the number of training examples,
        I is the dimension of the input and O is the dimension of the output.
        :return: Reference to the conditioned model
        """

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

        self.input_dim = xs.shape[1]
        self.output_dim = ys.shape[1]

        self.xs = xs
        self.ys = ys
        self.num_true_points = xs.shape[0]

        return self

    @abc.abstractmethod
    def train(self) -> None:
        pass

    @abc.abstractmethod
    def predict_batch(self, xs: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def normalize(a: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        return (a - mean) / std

    def get_prior_gp_model(self, length_scale, gp_variance, noise_variance):

        # Create a new Stheno graph for the GP. This step is crucial
        g = Graph()

        # Construct parameterized kernel
        kernel = self.AVAILABLE_KERNELS[self.kernel_name]()

        try:
            prior_gp = gp_variance * GP(kernel, graph=g).stretch(length_scale) + \
                       noise_variance * GP(Delta(), graph=g)

        except Exception as e:
            print("Creating GP prior failed: {}".format(str(e)))

            print(type(gp_variance))
            print(gp_variance)
            print("-")
            print(length_scale)
            print("-")
            print(noise_variance)

            raise e

        return prior_gp

    def _update_mean_std(self, min_std=1e-10) -> None:

        xs_mean, xs_var = tf.nn.moments(self.xs, axes=[0])

        self.xs_mean = xs_mean
        self.xs_std = tf.maximum(tf.sqrt(xs_var), min_std)

        ys_mean, ys_var = tf.nn.moments(self.ys, axes=[0])

        self.ys_mean = ys_mean
        self.ys_std = tf.maximum(tf.sqrt(ys_var), min_std)

    def add_true_point(self, x, y) -> None:

        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)

        if x.shape != (1, self.input_dim):
            raise ModelError("x with shape {} must have shape {}!".format(x.shape, (1, self.input_dim)))
        if y.shape != (1, self.output_dim):
            raise ModelError("y with shape {} must have shape {}!".format(y.shape, (1, self.output_dim)))

        assert self.num_pseudo_points == 0

        self._append_data_point(x, y)
        self.num_true_points += 1

    def add_pseudo_point(self, x):

        x = tf.convert_to_tensor(x, dtype=tf.float64)

        if x.shape != (1, self.input_dim):
            raise ModelError("point with shape {} must have shape {}!".format(x.shape, (1, self.input_dim)))

        mean, var = self.predict_batch(x)

        self._append_data_point(x, mean)
        self.num_pseudo_points += 1

    def remove_pseudo_points(self) -> None:
        self.xs = self.xs[:-self.num_pseudo_points, :]
        self.ys = self.ys[:-self.num_pseudo_points, :]
        self.num_pseudo_points = 0

    def _append_data_point(self, x, y) -> None:
        self.xs = tf.concat((self.xs, x), axis=0)
        self.ys = tf.concat((self.ys, y), axis=0)
