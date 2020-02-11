import abc
import tensorflow as tf
import numpy as np

# Hungarian algorithm
from scipy.optimize import linear_sum_assignment


class Distribution(tf.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dtype=tf.float64, name="distribution", **kwargs):
        super(Distribution, self).__init__(name=name, **kwargs)

        self.dtype = dtype

    @abc.abstractmethod
    @tf.Module.with_name_scope
    def sample(self):
        pass


class GumbelMatching(Distribution):
    def __init__(self, weight_matrix, dtype=tf.float64, name="gumbel_matching_distribution", **kwargs):
        super(GumbelMatching, self).__init__(dtype=dtype, name=name, **kwargs)

        self.weight_matrix = tf.convert_to_tensor(weight_matrix, dtype=self.dtype)

    @tf.Module.with_name_scope
    def sample(self, as_tuple=False):
        """
        :param as_tuple: If true, we return a permutation tuple equivalent to right-multiplying with the matrix.
        :return:
        """

        # Create Gumbel(0, 1) noise
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(shape=self.weight_matrix.shape, dtype=self.dtype)))

        perturbed_weights = self.weight_matrix + gumbel_noise

        # Get assignment indices
        x_ind, y_ind = linear_sum_assignment(perturbed_weights.numpy())

        assignment = np.zeros(perturbed_weights.shape)
        assignment[x_ind, y_ind] = 1

        assignment = tf.convert_to_tensor(assignment, dtype=self.dtype)

        if as_tuple:
            assignment = tuple(tf.argmax(assignment, axis=0).numpy())

        return assignment
