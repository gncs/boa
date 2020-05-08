import pytest

from boa.core.gp import GaussianProcess, CoreError

import numpy as np
import tensorflow as tf

import logging

# Disable logging for tests
logging.disable(logging.CRITICAL)

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


# Target function (noise free).
def f(x):
    return (np.sinc(3 * x) + 0.5 * (x - 0.5)**2).reshape(-1, 1)


@pytest.fixture
def data():
    # Generate X's and Y's for training.
    np.random.seed(42)

    xs = np.array([-0.25, 0, 0.1]).reshape(-1, 1)
    ys = f(xs)

    return xs, ys


@pytest.fixture
def gp():
    return GaussianProcess(kernel="rbf", input_dim=1, signal_amplitude=1, length_scales=1, noise_amplitude=0.01)


@pytest.mark.parametrize(
    'kernel, signal_amplitude, length_scales, noise_amplitude',
    [("bla", 1, 1, 0.01), ("rbf", -1, 1, 0.01), ("rbf", 1, -1, 0.01), ("rbf", 1, 1, -0.01),
     ("rbf", 1, np.array([1, 1, 0, 1, 1, 1]), 0.01), ("rbf", 1, 1, 0),
     ("rbf", 1, tf.ones((4, 4), dtype=tf.float64), 0.01)],
)
def test_init(kernel, signal_amplitude, length_scales, noise_amplitude):

    with pytest.raises(CoreError):
        gp = GaussianProcess(kernel=kernel,
                             input_dim=1,
                             signal_amplitude=signal_amplitude,
                             length_scales=length_scales,
                             noise_amplitude=noise_amplitude)


def test_copy(gp):

    # Copy the GP
    gp_copy = gp.copy()

    # Check equality of kernel parameters
    assert tf.reduce_all(tf.equal(gp_copy.kernel_name, gp.kernel_name))
    assert tf.reduce_all(tf.equal(gp_copy.gp_signal_amplitude, gp.gp_signal_amplitude))
    assert tf.reduce_all(tf.equal(gp_copy.gp_noise_amplitude, gp.gp_noise_amplitude))
    assert tf.reduce_all(tf.equal(gp_copy.gp_length_scales, gp.gp_length_scales))

    # Equality of assigned data
    assert tf.reduce_all(tf.equal(gp_copy.xs, gp.xs))
    assert tf.reduce_all(tf.equal(gp_copy.ys, gp.ys))

    # Copy must use a different Stheno graph
    assert gp_copy.graph != gp.graph


def test_conditioning(gp, data):

    conditioned_gp = gp | data

    # Test that the conditioned GP is not the same one we started with
    assert conditioned_gp != gp

    # Make sure the two gps are on separate Stheno graphs
    assert conditioned_gp.graph != gp.graph

    # Make sure the unconditioned GP still has no data to it
    assert gp.xs.shape[0] == 0
    assert gp.ys.shape[0] == 0


def test_gp_data_standardization():

    test_xs = tf.random.uniform(shape=(10, 3))
    test_ys = tf.random.uniform(shape=(10, 1))

    gp = GaussianProcess(kernel='rbf', input_dim=3, signal_amplitude=1., length_scales=1., noise_amplitude=0.1)

    mean, std = gp.xs_mean_and_std

    gp = gp | (test_xs[:5, :], test_ys[:5, :])

    print(gp.xs_mean_and_std)


def test_log_pdf():
    pass


def test_sample():
    pass


def test_predict():
    pass
