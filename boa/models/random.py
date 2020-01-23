import logging
import json
import tensorflow as tf

from .abstract_model import AbstractModel

from boa.core.utils import setup_logger

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/random_model.log")


class RandomModel(AbstractModel):
    def __init__(self, input_dim, output_dim, seed, num_samples, name="random_model", **kwargs):

        super(RandomModel, self).__init__(kernel="rbf", input_dim=input_dim, output_dim=output_dim, name=name, **kwargs)

        self.seed = seed
        self._internal_state = seed

        self.num_samples = num_samples

    def fit(self, **kwargs):
        if self.verbose:
            logger.info("Random model needs no fitting!")

    def predict(self, xs, numpy=False):

        xs = self._validate_and_convert(xs, output=False)

        ys_mean, ys_var = tf.nn.moments(self.ys, axes=[0], keepdims=True)

        # Always have some minimum variance
        ys_var = tf.maximum(ys_var, 1e-10)

        ys_std = tf.math.sqrt(ys_var)

        # Fix randomness and advance random state
        tf.random.set_seed(self._internal_state)
        self._internal_state += 1

        pred_mean = tf.random.normal(mean=ys_mean,
                                     stddev=ys_std,
                                     shape=(xs.shape[0], self.output_dim),
                                     dtype=tf.float64)

        pred_samples = tf.random.normal(mean=ys_mean,
                                        stddev=ys_std,
                                        shape=(self.num_samples, xs.shape[0], self.output_dim),
                                        dtype=tf.float64)

        _, pred_var = tf.nn.moments(pred_samples, axes=[0], keepdims=False)

        if numpy:
            pred_mean = pred_mean.numpy()
            pred_var = pred_var.numpy()

        return pred_mean, pred_var

    def get_config(self):

        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "verbose": self.verbose
        }

    @staticmethod
    def from_config(config):

        return RandomModel(**config)

    def create_gps(self):
        if self.verbose:
            logger.info("No GPs used for Random Model!")

    @staticmethod
    def restore(save_path):
        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = RandomModel.from_config(config)

        model.load_weights(save_path)

        return model
