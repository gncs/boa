import abc
import logging
import json

import tensorflow as tf

from boa.core.gp import GaussianProcess
from boa.core.utils import tensor_hash
from boa.core.utils import calculate_euclidean_distance_percentiles, calculate_per_dimension_distance_percentiles, \
    setup_logger

from not_tf_opt import BoundedVariable

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True)


class ModelError(Exception):
    """Base error thrown by models"""


class AbstractModel(tf.keras.Model, abc.ABC):
    AVAILABLE_LENGTHSCALE_INITIALIZATIONS = ["random", "l2_median", "marginal_median"]

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 kernel_args={},
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

        super(AbstractModel, self).__init__(name=name,
                                            dtype=tf.float64,
                                            **kwargs)

        self.models = []

        # Check if the specified kernel is available
        if kernel in GaussianProcess.AVAILABLE_KERNELS:
            self.kernel_name = kernel
            self.kernel_args = kernel_args
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

        # ---------------------------------------------------------------------
        # Model hyperparameters
        # ---------------------------------------------------------------------
        self.length_scales = []
        self.signal_amplitudes = []
        self.noise_amplitudes = []

        # ---------------------------------------------------------------------
        # Flags
        # ---------------------------------------------------------------------
        self.trained = tf.Variable(False, name="trained", trainable=False)

        # ---------------------------------------------------------------------
        # Statistics for initialization
        # ---------------------------------------------------------------------

        # Statistics on the inputs of each GP in our model
        self._gp_input_statistics = [None] * self.output_dim
        self._gp_input_statistics_hashes = [None] * self.output_dim

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

    def initialize_hyperparamters(self,
                                  index: int,
                                  length_scale_init_mode: str,
                                  random_init_lower_bound: float = 0.5,
                                  random_init_upper_bound: float = 2.0,
                                  length_scale_base_lower_bound: float = 1e-2,
                                  length_scale_base_upper_bound: float = 1e2,
                                  signal_lower_bound=1e-2,
                                  signal_upper_bound=1e1,
                                  noise_scale_factor=0.1,
                                  percentiles=(0, 10, 30, 50, 70, 90, 100)):
        """
        Creates the initializers for the length scales, signal amplitudes and noise variances.
        :param length_scale_init_mode:
        :param index:
        :return:
        """

        if length_scale_init_mode not in self.AVAILABLE_LENGTHSCALE_INITIALIZATIONS:
            raise ModelError(f"Length scale initialization mode must be one of "
                             f"{self.AVAILABLE_LENGTHSCALE_INITIALIZATIONS}! ({length_scale_init_mode} was given)")

        # Get the input data for the current GP
        gp_input = self.gp_input(index=index)

        # Dimension of a single training example
        gp_input_dim = gp_input.shape[1]

        # Check if we have already calculated the statistics for this input
        gp_input_hash = tensor_hash(gp_input)

        # If the data changed, we calculate the statistics again
        if gp_input_hash != self._gp_input_statistics_hashes[index]:
            # Set new hash
            self._gp_input_statistics_hashes[index] = gp_input_hash

            # Calculate new statistics
            l2_percentiles = calculate_euclidean_distance_percentiles(gp_input, percentiles)
            marginal_percentiles = calculate_per_dimension_distance_percentiles(gp_input, percentiles)

            # Cast them to the data type of the model
            l2_percentiles = tf.cast(l2_percentiles, self.dtype)
            marginal_percentiles = tf.cast(marginal_percentiles, self.dtype)

            # Set the new statistics
            self._gp_input_statistics[index] = {"l2": l2_percentiles,
                                                "dim": marginal_percentiles}

        # ---------------------------------------------------------------------
        # Random initialization
        # ---------------------------------------------------------------------
        if length_scale_init_mode == "random":
            ls_init = tf.random.uniform(shape=(gp_input_dim,),
                                        minval=random_init_lower_bound,
                                        maxval=random_init_upper_bound,
                                        dtype=tf.float64)

            ls_lower_bound = length_scale_base_lower_bound
            ls_upper_bound = length_scale_base_upper_bound

        # ---------------------------------------------------------------------
        # Initialization using the median of the non-zero pairwise Euclidean
        # distances between training inputs
        # ---------------------------------------------------------------------
        elif length_scale_init_mode == "l2_median":
            l2_percentiles = self._gp_input_statistics[index]["l2"]

            # Center on the medians
            ls_init = l2_percentiles[3]

            ls_rand_range = tf.minimum(l2_percentiles[3] - l2_percentiles[2],
                                       l2_percentiles[4] - l2_percentiles[3])

            ls_init += tf.random.uniform(shape=(gp_input_dim,),
                                         minval=-ls_rand_range,
                                         maxval=ls_rand_range,
                                         dtype=self.dtype)

            ls_lower_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_lower_bound = ls_lower_bound * l2_percentiles[0] / (4. * tf.sqrt(gp_input_dim))

            ls_upper_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_upper_bound = ls_upper_bound * l2_percentiles[-1] * 64. / tf.sqrt(gp_input_dim)

        # ---------------------------------------------------------------------
        # Initialization using the marginal median of pairwise distances
        # between training input dimensions
        # ---------------------------------------------------------------------
        elif length_scale_init_mode == "marginal_median":
            marginal_percentiles = self._gp_input_statistics[index]["marginal"]

            # Center on the medians
            ls_init = marginal_percentiles[3, :]

            ls_rand_range = tf.minimum(marginal_percentiles[3, :] - marginal_percentiles[2, :],
                                       marginal_percentiles[4, :] - marginal_percentiles[3, :])

            ls_init += tf.random.uniform(shape=(gp_input_dim,),
                                         minval=-ls_rand_range,
                                         maxval=ls_rand_range,
                                         dtype=self.dtype)

            # We need to multiply the lengthscales by sqrt(N) to correct for the number of dimensions
            dim_coeff = tf.sqrt(tf.cast(self.input_dim + index, tf.float64))
            ls_init = ls_init * dim_coeff

            ls_lower_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_lower_bound = ls_lower_bound * marginal_percentiles[0, :] / 4.

            ls_upper_bound = tf.ones(shape=(gp_input_dim,), dtype=self.dtype)
            ls_upper_bound = ls_upper_bound * marginal_percentiles[-1, :] * 64.

        else:
            raise NotImplementedError

        # Create bounded variables
        length_scales = BoundedVariable(ls_init,
                                        lower=tf.maximum(ls_lower_bound, length_scale_base_lower_bound),
                                        upper=tf.minimum(ls_upper_bound, length_scale_base_upper_bound),
                                        dtype=self.dtype)

        signal_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                             minval=random_init_lower_bound,
                                                             maxval=random_init_upper_bound,
                                                             dtype=self.dtype),
                                           lower=signal_lower_bound,
                                           upper=signal_upper_bound,
                                           dtype=self.dtype)

        noise_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                            minval=noise_scale_factor * random_init_lower_bound,
                                                            maxval=noise_scale_factor * random_init_upper_bound,
                                                            dtype=self.dtype),
                                          lower=noise_scale_factor * signal_lower_bound,
                                          upper=noise_scale_factor * signal_upper_bound,
                                          dtype=self.dtype)

        return length_scales, signal_amplitude, noise_amplitude

    @abc.abstractmethod
    def fit(self,
            optimizer="l-bfgs-b",
            optimizer_restarts=1,
            iters=1000,
            tolerance=1e-5,
            trace=False,
            seed=None,
            **kwargs) -> None:

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)


    @abc.abstractmethod
    def gp_input(self, index):
        pass

    @abc.abstractmethod
    def predict(self, xs, numpy=False, **kwargs):
        pass

    @abc.abstractmethod
    def log_prob(self, xs, ys, use_conditioning_data=True, numpy=False):
        pass

    @abc.abstractmethod
    def create_data_getter(self, xs, ys):
        """
        Closure that returns a function that can be called with an index, and will return
        a tuple (xs_i, ys_i) - the input, output pairs of the ith GP.

        :param xs:
        :param ys:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_config(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_config(config, **kwargs):
        pass

    @abc.abstractmethod
    def create_gps(self):
        pass

    def save(self, save_path, **kwargs):

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

        xs = tf.convert_to_tensor(xs)
        xs = tf.cast(xs, tf.float64)

        if len(xs.shape) == 1:
            second_dim = self.output_dim if output else self.input_dim

            # Attempt to convert the xs to the right shape
            xs = tf.reshape(xs, (-1, second_dim))

        # Check if the shapes are correct
        if not len(xs.shape) == 2:
            raise ModelError("The input must be of rank 2!")

        if (not output and xs.shape[1] != self.input_dim) or \
                (output and xs.shape[1] != self.output_dim):
            out_text = 'output' if output else 'input'
            raise ModelError(f"The second dimension of the {out_text} "
                             f"is incorrect: {xs.shape[1]} (expected {self.output_dim if output else self.input_dim})!")

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
        percentiles = [0, 10, 30, 50, 70, 90, 100]

        self.xs_euclidean_percentiles = tf.cast(calculate_euclidean_distance_percentiles(xs, percentiles), self.dtype)
        self.ys_euclidean_percentiles = tf.cast(calculate_euclidean_distance_percentiles(ys, percentiles), self.dtype)

        self.xs_per_dim_percentiles = tf.cast(calculate_per_dimension_distance_percentiles(xs, percentiles), self.dtype)
        self.ys_per_dim_percentiles = tf.cast(calculate_per_dimension_distance_percentiles(ys, percentiles), self.dtype)
