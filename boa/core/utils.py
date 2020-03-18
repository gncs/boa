import tensorflow as tf
import tensorflow_probability as tfp
import logging

import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


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

        df[label] = df[label].apply(transform)

    return df


def back_transform(data, labels, transforms):

    if len(data.shape) == 1:
        data = data.reshape([-1, 1])

    if data.shape[1] != len(labels):
        raise CoreError(f"Number of label dimensions ({data.shape[1]}), "
                        f"must match length of label list ({len(labels)} given!")

    for i, (col, label) in enumerate(zip(data.T, labels)):

        if label in transforms:
            if transforms[label] == 'log':
                transform = np.exp

        else:
            transform = lambda x: x

        data[:, i] = transform(col)

    return data


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


def normalize(x, min_std=1e-10):

    # Calculate data statistics
    mean, var = tf.nn.moments(x, axes=[0], keepdims=True)
    std = tf.maximum(tf.sqrt(var), min_std)

    return (x - mean) / std


def distance_matrix(xs):
    """
    Calculate the pairwise distances between the rows of the input data-points and
    return the square matrix D_ij = || X_i - X_j ||

    Uses the identity that
    D_ij^2 = (X_i - X_j)'(X_i - X_j)
           = X_i'X_i - 2 X_i'X_j + X_j'X_j

    which can be computed efficiently using some nice broadcasting.
    """

    xs = normalize(xs)

    # Calculate L2 norm of each row in the matrix
    norms = tf.reduce_sum(xs * xs, axis=1, keepdims=True)

    cross_terms = -2 * tf.matmul(xs, xs, transpose_b=True)

    dist_matrix = norms + cross_terms + tf.transpose(norms)

    return dist_matrix


def dim_distance_matrix(xs):

    dist_mats = []

    xs = normalize(xs)

    for k in range(xs.shape[1]):

        # Select appropriate column from the matrix
        c = xs[:, k:k + 1]

        norms = c * c

        cross_terms = -2 * tf.matmul(c, c, transpose_b=True)

        dist_mat = norms + cross_terms + tf.transpose(norms)

        dist_mats.append(dist_mat)

    return tf.stack(dist_mats)


def calculate_euclidean_distance_percentiles(xs, percents):

    euclidean_dist_mat = distance_matrix(xs)

    # Filter 0s
    euclidean_dists = tf.gather_nd(euclidean_dist_mat, tf.where(euclidean_dist_mat != 0.))

    return tfp.stats.percentile(euclidean_dists, percents, axis=0)


def calculate_per_dimension_distance_percentiles(xs, percents):

    dim_dist_mat = dim_distance_matrix(xs)
    dim_dist_mat = tf.reshape(dim_dist_mat, (xs.shape[1], -1))

    dim_percentiles = []

    # Filter 0s and calculate percentiles per dimension
    for i, row in enumerate(dim_dist_mat):
        dim_dists = tf.gather_nd(row, tf.where(row != 0.))

        if dim_dists.shape[0] > 0:
            dim_percentile = tfp.stats.percentile(dim_dists, percents, axis=0)
        else:
            dim_percentile = tf.zeros((len(percents), ), dtype=xs.dtype)

        dim_percentiles.append(dim_percentile)

    return tf.stack(dim_percentiles)
