from unittest import TestCase

from boa.models.multi_output_gp_regression_model import MultiOutputGPRegressionModel

import tensorflow as tf

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


class TestAbstractModel(TestCase):
    def test_init(self):
        instance = MultiOutputGPRegressionModel(kernel="rbf", input_dim=1, output_dim=1)
