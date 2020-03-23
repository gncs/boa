import logging

import json

from typing import List

import tensorflow as tf
import numpy as np

from boa.core.gp import GaussianProcess
from boa.core.utils import setup_logger
from .abstract_model import AbstractModel, ModelError
from boa import ROOT_DIR

from not_tf_opt import minimize, BoundedVariable

__all__ = ["FullyFactorizedGPModel"]

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file=f"{ROOT_DIR}/../logs/ff_gp.log")


class FullyFactorizedGPModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 initialization_heuristic: str = "median",
                 parallel: bool = False,
                 name: str = "gp_model",
                 verbose=True,
                 **kwargs):

        super(FullyFactorizedGPModel, self).__init__(kernel=kernel,
                                                     input_dim=input_dim,
                                                     output_dim=output_dim,
                                                     parallel=parallel,
                                                     verbose=verbose,
                                                     name=name,
                                                     **kwargs)

        self.initialization_heuristic = initialization_heuristic

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            self.length_scales.append(
                tf.Variable(tf.ones(self.input_dim, dtype=tf.float64), name=f"{i}/length_scales", trainable=False))

            self.signal_amplitudes.append(
                tf.Variable((1.0,), dtype=tf.float64, name=f"{i}/signal_amplitude", trainable=False))

            self.noise_amplitudes.append(
                tf.Variable((1.0,), dtype=tf.float64, name=f"{i}/noise_amplitude", trainable=False))

    def gp_input(self, index):
        return self.xs

    def gp_input_dim(self, index):
        return self.input_dim

    def gp_output(self, index):
        return self.ys[:, index:index + 1]

    def predict(self, xs, numpy=False):

        if not self.trained:
            logger.warning("Using untrained model for prediction!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs = self._validate_and_convert(xs, output=False)

        means = []
        variances = []

        for i, model in enumerate(self.models):
            model = model | (self.xs, self.ys[:, i:i + 1])

            mean, var = model.predict(xs, latent=False)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances

    def log_prob(self, xs, ys, use_conditioning_data=True, latent=True, numpy=False, target_dims=None):

        if target_dims is not None and not isinstance(target_dims, (tuple, list)):
            raise ModelError("target_dims must be a list or a tuple!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        log_prob = 0.

        for i, model in enumerate(self.models):

            if i not in target_dims:
                continue

            cond_model = model | (self.xs, self.ys[:, i:i + 1])

            if use_conditioning_data:
                model_log_prob = cond_model.log_pdf(xs,
                                                    ys[:, i:i + 1],
                                                    latent=latent,
                                                    with_jitter=False,
                                                    normalize_with_training_data=True)
            else:
                # Normalize model to the regime on which the models were trained
                norm_xs = cond_model.normalize_with_training_data(xs, output=False)
                norm_ys = cond_model.normalize_with_training_data(ys[:, i:i + 1], output=True)

                model_log_prob = model.log_pdf(norm_xs,
                                               norm_ys,
                                               latent=latent,
                                               with_jitter=False)

            log_prob = log_prob + model_log_prob

        if numpy:
            log_prob = log_prob.numpy()

        return log_prob

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = FullyFactorizedGPModel.from_config(config, )

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):

        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "initialization_heuristic": self.initialization_heuristic,
            "parallel": self.parallel,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, **kwargs):
        return FullyFactorizedGPModel(**config)
