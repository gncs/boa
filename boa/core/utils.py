import tensorflow as tf
import tensorflow_probability as tfp
import logging


class CoreError(Exception):
    """
    Base error thrown by modules in the core
    """


def tf_bounded_variable(init, lower, upper, name=None, dtype=tf.float64):

    init = tf.convert_to_tensor(init, dtype=dtype)

    # Calculate the reparametrized value first
    var_init = (init - lower) / (upper - lower)

    if name is not None:
        var = tf.Variable(var_init, name=name)
    else:
        var = tf.Variable(var_init)

    def transform(x):
        return (upper - lower) * var + lower

    def assign(x):
        x = tf.convert_to_tensor(x, dtype=dtype)
        x = tf.reshape(x, var.shape)
        var.assign((x - lower) / (upper - lower + 1e-12))

    return var, transform, assign


def setup_logger(name, level, log_file=None, to_console=False, format="%(levelname)s:%(name)s:%(message)s"):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(format)

    if log_file is not None:

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    if to_console:

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        #logger.addHandler(stream_handler)

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
            dim_percentile = tf.zeros((len(percents),), dtype=xs.dtype)

        dim_percentiles.append(dim_percentile)

    return tf.stack(dim_percentiles)
