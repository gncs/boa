import abc
import logging

import tensorflow as tf
import tensorflow_probability as tfp

from boa.core.gp import GaussianProcess
from boa.core.utils import calculate_euclidean_distance_percentiles, calculate_per_dimension_distance_percentiles, setup_logger

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True)


class ModelError(Exception):
    """Base error thrown by models"""


class AbstractModel(tf.keras.Model):

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

        self.models = []

        # Check if the specified kernel is available
        if kernel in GaussianProcess.AVAILABLE_KERNELS:
            self.kernel_name = kernel
        else:
            raise ModelError("Specified kernel {} not available!".format(kernel))

        self.num_optimizer_restarts = num_optimizer_restarts
        self.parallel = parallel

        self.input_dim = 0
        self.output_dim = 0

        self.verbose = verbose

        self.xs = None
        self.ys = None

        self.pairwise_distances = None
        self.pairwise_dim_distances = None

        self.dim_length_medians = None

        self.num_pseudo_points = 0
        self.num_true_points = 0

        # Range for the initialisation of GP hyperparameters
        self.init_minval = tf.constant(init_minval, dtype=tf.float64)
        self.init_maxval = tf.constant(init_maxval, dtype=tf.float64)

    @abc.abstractmethod
    def _set_data(self, xs, ys):
        """
        Adds data to the model. The notation is supposed to imitate
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param xs: rank-2 tensor: N x I where N is the number of training examples,
        I is the dimension of the input.
        :param ys: rank-2 tensor: N x O, where N is the number of training examples,
        O is the dimension of the output.

        :return: Reference to the conditioned model
        """

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

        # ---------------------------------------------
        # Calculate stuff for the median heuristic
        # ---------------------------------------------

        percentiles = [10, 30, 50, 70, 90]

        self.xs_euclidean_percentiles = calculate_euclidean_distance_percentiles(self.xs, percentiles)
        self.ys_euclidean_percentiles = calculate_euclidean_distance_percentiles(self.ys, percentiles)
        logging.debug(f"Input Euclidean distance percentiles (10, 30, 50, 70, 90): {self.xs_euclidean_percentiles}")
        logging.debug(f"Output Euclidean distance percentiles (10, 30, 50, 70, 90): {self.ys_euclidean_percentiles}")

        self.xs_per_dim_percentiles = calculate_per_dimension_distance_percentiles(self.xs, percentiles)
        self.ys_per_dim_percentiles = calculate_per_dimension_distance_percentiles(self.ys, percentiles)

    @abc.abstractmethod
    def fit(self, xs, ys) -> None:
        pass

    @abc.abstractmethod
    def predict_batch(self, xs):
        pass

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
