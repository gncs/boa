import abc
import logging
import json
import os

import tensorflow as tf

from boa.core.gp import GaussianProcess
from boa.core.utils import calculate_euclidean_distance_percentiles, calculate_per_dimension_distance_percentiles, setup_logger

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True)


class ModelError(Exception):
    """Base error thrown by models"""


class AbstractModel(tf.keras.Model):

    __metaclass__ = abc.ABCMeta

    XS_NAME = "inputs"
    YS_NAME = "outputs"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 parallel: bool = False,
                 verbose: bool = False,
                 _num_starting_data_points: int = 0,
                 name: str = "abstract_model", **kwargs):
        """

        :param kernel:
        :param input_dim:
        :param output_dim:
        :param parallel:
        :param verbose:
        :param _num_starting_data_points: Should not be set by the user. Only used to restore models.
        :param name:
        :param kwargs:
        """

        super(AbstractModel, self).__init__(name=name, **kwargs)

        self.models = []

        # Check if the specified kernel is available
        if kernel in GaussianProcess.AVAILABLE_KERNELS:
            self.kernel_name = kernel
        else:
            raise ModelError("Specified kernel {} not available!".format(kernel))

        self.parallel = parallel

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.verbose = verbose

        self.xs = tf.Variable(tf.zeros((_num_starting_data_points, input_dim), dtype=tf.float64), name=self.XS_NAME, trainable=False)
        self.ys = tf.Variable(tf.zeros((_num_starting_data_points, output_dim), dtype=tf.float64), name=self.YS_NAME, trainable=False)

        self.num_pseudo_points = 0
        self.num_true_points = 0

        self.xs_per_dim_percentiles = None
        self.ys_per_dim_percentiles = None

        self.xs_euclidean_percentiles = None
        self.ys_euclidean_percentiles = None

        self.trained = tf.Variable(False, name="trained", trainable=False)

    @abc.abstractmethod
    def copy(self, name=None):
        pass

    def condition_on(self, xs, ys):
        """
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param xs: rank-2 tensor: N x I where N is the number of training examples,
        I is the dimension of the input.
        :param ys: rank-2 tensor: N x O, where N is the number of training examples,
        O is the dimension of the output.

        :return: Reference to the conditioned model
        """

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        model = self.copy()

        xs = tf.concat((self.xs, xs), axis=0)

        model.xs = tf.Variable(xs, name=self.XS_NAME, trainable=False)
        model.ys = tf.Variable(ys, name=self.YS_NAME, trainable=False)
        model.num_true_points = xs.shape[0]

        return model


    @abc.abstractmethod
    def fit(self, xs, ys, optimizer_restarts=1) -> None:
        pass

    @abc.abstractmethod
    def predict(self, xs):
        pass

    @abc.abstractmethod
    def get_config(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_config(config, restore_num_data_points=False):
        pass

    @abc.abstractmethod
    def create_gps(self):
        pass

    def save(self, save_path):

        if not self.trained:
            logger.warning("Saved model has not been trained yet!")

        self.save_weights(save_path)

        config = self.get_config()

        with open(save_path + ".json", "w") as config_file:
            json.dump(config, config_file, indent=4, sort_keys=True)

    @staticmethod
    @abc.abstractmethod
    def restore(save_path):
        pass

    def _validate_and_convert_input_output(self, xs, ys):
        # Reasonable test of whether the inputs are array-like
        if not hasattr(xs, "__len__") or not hasattr(ys, "__len__"):
            raise ModelError("xs and ys must be array-like!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)
        ys = tf.convert_to_tensor(ys, dtype=tf.float64)

        # Check if the shapes are correct
        if not len(xs.shape) == 2 or not len(ys.shape) == 2:
            raise ModelError("The input and output must be of rank 2!")

        # Ensure the user provided the same number of input and output points
        if not xs.shape[0] == ys.shape[0]:
            raise ModelError("The first dimension of the input and the output must be equal! "
                             "(the data needs to form valid input-output pairs)")

        if not xs.shape[1] == self.input_dim:
            raise ModelError(f"The second dimension of the input must equal the set input dimension ({self.input_dim})!")

        if not ys.shape[1] == self.output_dim:
            raise ModelError(f"The second dimension of the output must equal the set output dimension ({self.output_dim})!")

        return xs, ys

    def _calculate_statistics_for_median_initialization_heuristic(self, xs, ys):

        # ---------------------------------------------
        # Calculate stuff for the median heuristic
        # ---------------------------------------------
        percentiles = [10, 30, 50, 70, 90]

        self.xs_euclidean_percentiles = calculate_euclidean_distance_percentiles(xs, percentiles)
        self.ys_euclidean_percentiles = calculate_euclidean_distance_percentiles(ys, percentiles)
        logging.debug(f"Input Euclidean distance percentiles (10, 30, 50, 70, 90): {self.xs_euclidean_percentiles}")
        logging.debug(f"Output Euclidean distance percentiles (10, 30, 50, 70, 90): {self.ys_euclidean_percentiles}")

        self.xs_per_dim_percentiles = calculate_per_dimension_distance_percentiles(xs, percentiles)
        self.ys_per_dim_percentiles = calculate_per_dimension_distance_percentiles(ys, percentiles)