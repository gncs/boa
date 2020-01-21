from unittest import TestCase

from boa.core.gp import GaussianProcess, CoreError
import numpy as np

import tensorflow as tf

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


class TestGaussianProcess(TestCase):
    def setUp(self) -> None:
        """
        We create a small dataset here
        """

        # Target function (noise free).
        def f(x):
            return (np.sinc(3 * x) + 0.5 * (x - 0.5)**2).reshape(-1, 1)

        # Generate X's and Y's for training.
        np.random.seed(42)

        self.xs = np.array([-0.25, 0, 0.1]).reshape(-1, 1)
        self.ys = f(self.xs)

        self.gp = GaussianProcess(kernel="rbf", signal_amplitude=1, length_scales=1, noise_amplitude=0.01)

    def test_init(self):

        # ----------------------------------------------------------------
        # Kernel must be in GaussianProcess.AVAILABLE_KERNELS
        # ----------------------------------------------------------------
        with self.assertRaises(CoreError):

            gp = GaussianProcess(kernel="bla", signal_amplitude=1, length_scales=1, noise_amplitude=0.01)

        # ----------------------------------------------------------------
        # Kernel parameters must be strictly positive
        # ----------------------------------------------------------------
        with self.assertRaises(CoreError):
            # signal amplitude negative
            gp = GaussianProcess(kernel="rbf", signal_amplitude=-1, length_scales=1, noise_amplitude=0.01)

        with self.assertRaises(CoreError):
            # length_scales negative
            gp = GaussianProcess(kernel="rbf", signal_amplitude=1, length_scales=-1, noise_amplitude=0.01)

        with self.assertRaises(CoreError):
            # noise amplitude negative
            gp = GaussianProcess(kernel="rbf", signal_amplitude=1, length_scales=1, noise_amplitude=-0.01)

        with self.assertRaises(CoreError):
            # signal amplitude zero
            gp = GaussianProcess(kernel="rbf", signal_amplitude=0, length_scales=1, noise_amplitude=0.01)

        with self.assertRaises(CoreError):
            # one length scales zero
            length_scales = np.ones(6)
            length_scales[3] = 0

            gp = GaussianProcess(kernel="rbf", signal_amplitude=1, length_scales=length_scales, noise_amplitude=0.01)

        with self.assertRaises(CoreError):
            # noise amplitude zero
            gp = GaussianProcess(kernel="rbf", signal_amplitude=1, length_scales=1, noise_amplitude=0)

        with self.assertRaises(CoreError):
            # jitter amplitude zero
            gp = GaussianProcess(kernel="rbf", signal_amplitude=1, length_scales=1, noise_amplitude=0.01, jitter=0)

        # ----------------------------------------------------------------
        # Lengths scales must be at most rank-1
        # ----------------------------------------------------------------
        with self.assertRaises(CoreError):
            gp = GaussianProcess(kernel="rbf",
                                 signal_amplitude=1,
                                 length_scales=tf.ones((4, 4), dtype=tf.float64),
                                 noise_amplitude=1)

    def test_copy(self):

        # Copy the GP
        gp = self.gp.copy()

        # Check equality of kernel parameters
        self.assertTrue(tf.reduce_all(tf.equal(self.gp.kernel_name, gp.kernel_name)))
        self.assertTrue(tf.reduce_all(tf.equal(self.gp.signal_amplitude, gp.signal_amplitude)))
        self.assertTrue(tf.reduce_all(tf.equal(self.gp.noise_amplitude, gp.noise_amplitude)))
        self.assertTrue(tf.reduce_all(tf.equal(self.gp.length_scales, gp.length_scales)))

        # Equality of assigned data
        self.assertTrue(tf.reduce_all(tf.equal(self.gp.xs, gp.xs)))
        self.assertTrue(tf.reduce_all(tf.equal(self.gp.ys, gp.ys)))

        # Copy must use a different Stheno graph
        self.assertNotEqual(self.gp.graph, gp.graph)

    def test_conditioning(self):
        conditioned_gp = self.gp | (self.xs, self.ys)

        # Test that the conditioned GP is not the same one we started with
        self.assertNotEqual(conditioned_gp, self.gp)

        # Make sure the two gps are on separate Stheno graphs
        self.assertNotEqual(conditioned_gp.graph, self.gp.graph)

        # Make sure the unconditioned GP still has no data to it
        self.assertEqual(self.gp.xs.shape[0], 0)
        self.assertEqual(self.gp.ys.shape[0], 0)

    def test_log_pdf(self):
        pass

    def test_sample(self):
        pass

    def test_predict(self):
        pass
