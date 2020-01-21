from unittest import TestCase

from boa.models.abstract_model import AbstractModel

import tensorflow as tf

# Set CPU as available physical device
tf.config.experimental.set_visible_devices([], 'GPU')


class TestAbstractModel(TestCase):
    def test_init(self):
        instance = AbstractModel(kernel="rbf", input_dim=1, output_dim=1)
