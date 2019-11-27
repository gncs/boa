import abc

import numpy as np
import tensorflow as tf

from boa.core.gp import GaussianProcess


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
    def _set_data(self, xs, ys) -> tf.keras.Model:
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

        # self.pairwise_distances = self.distance_matrix()
        # self.pairwise_dim_distances = self.dim_distance_matrix()
        # self.dim_length_medians = tfp.stats.percentile(
        #     tf.reshape(self.pairwise_dim_distances, (self.input_dim, -1)), 50, axis=1)

        return self

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

    def distance_matrix(self):
        """
        Calculate the pairwise distances between the rows of the input data-points and
        return the square matrix D_ij = || X_i - X_j ||

        Uses the identity that
        D_ij^2 = (X_i - X_j)'(X_i - X_j)
               = X_i'X_i - 2 X_i'X_j + X_j'X_j

        which can be computed efficiently using some nice broadcasting.
        """

        # Calculate L2 norm of each row in the matrix
        norms = tf.reduce_sum(self.xs * self.xs, axis=1, keepdims=True)

        cross_terms = -2 * tf.matmul(self.xs, self.xs, transpose_b=True)

        dist_matrix = norms + cross_terms + tf.transpose(norms)

        return tf.sqrt(dist_matrix)

    def dim_distance_matrix(self):

        dist_mats = []

        x_normalized = self.normalize(self.xs, self.xs_mean, self.xs_std)

        for k in range(self.input_dim):

            # Select appropriate column from the matrix
            c = x_normalized[:, k:k + 1]

            norms = c * c

            cross_terms = -2 * tf.matmul(c, c, transpose_b=True)

            dist_mat = norms + cross_terms + tf.transpose(norms)

            dist_mats.append(dist_mat)

        return tf.sqrt(tf.stack(dist_mats))
