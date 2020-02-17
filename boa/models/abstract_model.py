import abc
import logging
import json
import os

import tensorflow as tf

from boa.core.gp import GaussianProcess
from boa.core.utils import calculate_euclidean_distance_percentiles, calculate_per_dimension_distance_percentiles, \
    setup_logger

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True)


class ModelError(Exception):
    """Base error thrown by models"""


class AbstractModel(tf.keras.Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 parallel: bool = False,
                 verbose: bool = False,
                 name: str = "abstract_model",
                 **kwargs):
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

        self.xs = tf.Variable(tf.zeros((0, input_dim), dtype=tf.float64),
                              name="inputs",
                              trainable=False,
                              shape=(None, input_dim))

        self.ys = tf.Variable(tf.zeros((0, output_dim), dtype=tf.float64),
                              name="outputs",
                              trainable=False,
                              shape=(None, output_dim))

        self.xs_per_dim_percentiles = None
        self.ys_per_dim_percentiles = None

        self.xs_euclidean_percentiles = None
        self.ys_euclidean_percentiles = None

        self.trained = tf.Variable(False, name="trained", trainable=False)

    def copy(self, name=None):

        # Reflect the class of the current instance
        constructor = self.__class__

        # Get the config of the instance
        config = self.get_config()

        # Instantiate the model
        model = constructor(**config)

        # Create dictionaries of model variables
        self_dict = {v.name: v for v in self.variables}
        model_dict = {v.name: v for v in model.variables}

        # Copy variables over
        for k, v in self_dict.items():
            model_dict[k].assign(v)

        return model

    def condition_on(self, xs, ys, keep_previous=True):
        """
        the conditioning operation:

        posterior = prior | (xs, ys)

        :param xs: rank-2 tensor: N x I where N is the number of training examples,
        I is the dimension of the input.
        :param ys: rank-2 tensor: N x O, where N is the number of training examples,
        O is the dimension of the output.
        :param keep_previous: if True, the data on which we conditioned before is retained as well.

        :return: Reference to the conditioned model
        """

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        model = self.copy()

        if keep_previous:
            xs = tf.concat((self.xs, xs), axis=0)
            ys = tf.concat((self.ys, ys), axis=0)

        model.xs.assign(xs)
        model.ys.assign(ys)

        return model

    def fit_to_conditioning_data(self, **kwargs):
        """

        :param kwargs: keyword arguments to be passed to optimize()
        :return:
        """
        self.fit(xs=self.xs.value(), ys=self.ys.value(), **kwargs)

    @abc.abstractmethod
    def fit(self, xs, ys, optimizer="l-bfgs-b", optimizer_restarts=1) -> None:
        pass

    @abc.abstractmethod
    def predict(self, xs, numpy=False):
        pass

    @abc.abstractmethod
    def log_prob(self, xs, ys, use_conditioning_data=True, numpy=False):
        pass

    @abc.abstractmethod
    def get_config(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_config(config):
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

    def _validate_and_convert(self, xs, output=False):

        # Reasonable test of whether the inputs are array-like
        if not hasattr(xs, "__len__"):
            raise ModelError("input must be array-like!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        # Convert a vector to "row vector"
        if len(xs.shape) == 1:
            xs = tf.reshape(xs, (1, -1))

        # Check if the shapes are correct
        if not len(xs.shape) == 2:
            raise ModelError("The input must be of rank 2!")

        if (not output and xs.shape[1] != self.input_dim) or \
                (output and xs.shape[1] != self.output_dim):
            raise ModelError(f"The second dimension of the input is incorrect: {xs.shape[1]}!")

        return xs

    def _validate_and_convert_input_output(self, xs, ys):

        xs = self._validate_and_convert(xs, output=False)
        ys = self._validate_and_convert(ys, output=True)

        # Ensure the user provided the same number of input and output points
        if not xs.shape[0] == ys.shape[0]:
            raise ModelError(f"The first dimension of the input ({xs.shape[0]}) and the output ({ys.shape[0]}) must "
                             f"be equal! (the data needs to form valid input-output pairs)")

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
