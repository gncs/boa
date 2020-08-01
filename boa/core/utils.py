import json
import functools
import tensorflow as tf
import tensorflow_probability as tfp
import logging

import numpy as np

from typing import Iterable, NamedTuple, Union, List, Tuple, Callable
from not_tf_opt import AbstractVariable

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


class InputSpec(NamedTuple):
    name: str
    domain: Union[Tuple, List, np.array, tf.Tensor]
    formula: Callable = (lambda x: x)


class CoreError(Exception):
    """
    Base error thrown by modules in the core
    """


def transform_df(df, transforms):
    for label, tr in transforms.items():

        if tr == 'log':
            transform = np.log

        else:
            transform = lambda x: x

        transformed = df[label].apply(transform)
        df.loc[:, label] = transformed

    return df


def back_transform(mean, variance, labels, transforms):
    if len(mean.shape) == 1:
        mean = mean.reshape([-1, 1])

    if len(variance.shape) == 1:
        variance = variance.reshape([-1, 1])

    if mean.shape[1] != len(labels):
        raise CoreError(f"Number of label dimensions ({mean.shape[1]}), "
                        f"must match length of label list ({len(labels)} given!")

    for i, (m, v, label) in enumerate(zip(mean.T, variance.T, labels)):

        if label in transforms:
            if transforms[label] == 'log':
                # We need to transform according to a log-Normal's statistics
                new_mean = np.exp(m)
                new_variance = (np.exp(v) - 1) * np.exp(2 * m + v)

                mean[:, i] = new_mean
                variance[:, i] = new_variance

    return mean, variance


def inv_perm(perm):
    perm = tf.convert_to_tensor(perm, dtype=tf.int32)
    return tf.scatter_nd(indices=tf.reshape(perm, [-1, 1]), updates=tf.range(tf.size(perm)), shape=perm.shape)


def sigmoid_inverse(x):
    if tf.reduce_any(x < 0.) or tf.reduce_any(x > 1.):
        raise ValueError(f"x = {x} was not in the sigmoid function's range ([0, 1])!")
    x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)

    return -tf.math.log(1. / x - 1.)


def setup_logger(name,
                 level,
                 log_file=None,
                 to_console=False,
                 format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                 datefmt='%d/%m/%Y %I:%M:%S %p'):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(format, datefmt=datefmt)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        logger.addHandler(file_handler)

    if to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)

        logger.addHandler(stream_handler)

    return logger


def standardize(x, min_std=1e-10):

    mean, std = get_mean_and_std(x, min_std)

    return (x - mean) / std


def get_mean_and_std(x, min_std=1e-10):
    # Calculate data statistics
    mean, var = tf.nn.moments(x, axes=[0], keepdims=True)
    std = tf.maximum(tf.sqrt(var), min_std)

    return mean, std

def distance_matrix(xs, eps=1e-12):
    """
    Calculate the pairwise distances between the rows of the input data-points and
    return the square matrix D_ij = || X_i - X_j ||

    Uses the identity that
    D_ij^2 = (X_i - X_j)'(X_i - X_j)
           = X_i'X_i - 2 X_i'X_j + X_j'X_j

    which can be computed efficiently using some nice broadcasting.
    """

    # print("unstandardized", xs)
    xs = standardize(xs)
    # print("std", xs)

    # Calculate L2 norm of each row in the matrix
    norms = tf.reduce_sum(xs * xs, axis=1, keepdims=True)

    cross_terms = -2 * tf.matmul(xs, xs, transpose_b=True)

    dist_matrix = norms + cross_terms + tf.transpose(norms)

    return tf.sqrt(dist_matrix + eps)


def dim_distance_matrix(xs):
    xs = standardize(xs)

    diffs = tf.abs(xs[None, :, :] - xs[:, None, :])

    return diffs


def calculate_euclidean_distance_percentiles(xs, percents, eps=1e-4):
    euclidean_dist_mat = distance_matrix(xs)

    # print("ED shape", euclidean_dist_mat.shape)
    # print("ED", euclidean_dist_mat.numpy())
    # Remove very small entries
    positive_euclidean_dists = tf.gather_nd(euclidean_dist_mat, tf.where(euclidean_dist_mat > eps))

    # print("PED shape", positive_euclidean_dists.shape)
    # print("PED", positive_euclidean_dists.numpy())

    try:
        percentiles = tfp.stats.percentile(positive_euclidean_dists, percents, axis=0)
    except Exception as e:
        print(str(e))

    return percentiles


def calculate_per_dimension_distance_percentiles(xs, percents, eps=1e-4):
    dim_dist_mat = dim_distance_matrix(xs)

    dim_percentiles = []

    for i in range(dim_dist_mat.shape[-1]):
        # Remove very small entries
        positive_dists = tf.gather_nd(dim_dist_mat[:, :, i], tf.where(dim_dist_mat[:, :, i] > eps))

        dim_percentiles.append(tfp.stats.percentile(positive_dists, percents))

    return tf.stack(dim_percentiles, axis=1)


def tensor_hash(tensor):
    """
    Hashes a tensorflow tensor based on its values
    :param tensor:
    :return:
    """

    if isinstance(tensor, (tf.Tensor, tf.Variable, AbstractVariable)):
        if isinstance(tensor, AbstractVariable):
            tensor = tensor()

        return tf.py_function(lambda s: hash(s.numpy().tostring()), inp=[tensor], Tout=tf.int64)

    elif isinstance(tensor, Iterable):
        return sum([tensor_hash(t) for t in tensor])

    else:
        raise CoreError(f"tensor must be an iterable or a TF tensor, but had type {type(tensor)}!")


def tf_custom_gradient_method(f):
    """
    Allows to declare a class method to have custom gradients.
    Taken from: https://stackoverflow.com/questions/54819947/defining-custom-gradient-as-a-class-method-in-tensorflow
    :param f:
    :return:
    """
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, '_tf_custom_gradient_wrappers'):
            self._tf_custom_gradient_wrappers = {}
        if f not in self._tf_custom_gradient_wrappers:
            self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a, **kw: f(self, *a, **kw))
        return self._tf_custom_gradient_wrappers[f](*args, **kwargs)
    return wrapped

class NumpyEncoder(json.JSONEncoder):
   def default(self, obj):
       if isinstance(obj, np.integer):
           return int(obj)
       elif isinstance(obj, np.floating):
           return float(obj)
       elif isinstance(obj, np.ndarray):
           return obj.tolist()
       else:
           return super(NumpyEncoder, self).default(obj)

