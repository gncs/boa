import logging

from typing import List

import json

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from boa.core.gp import DiscreteGaussianProcess
from boa.core.utils import setup_logger
from .abstract_model import AbstractModel

from boa.core.kernel import perm_pointwise_distance

from boa.core.variables import BoundedVariable
from boa.core.optimize import bounded_minimize

__all__ = ["PermutationGPModel"]

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/permutation_gp.log")


class PermutationGPModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 distance_kind: str,
                 input_dim: int,
                 name: str = "permutation_gp_model",
                 **kwargs):
        super().__init__(kernel=kernel,
                         kernel_args={"kind": distance_kind},
                         input_dim=input_dim,
                         output_dim=1,
                         parallel=False,
                         verbose=False,
                         name=name,
                         **kwargs)

        self.distance_kind = distance_kind

        self.length_scale = tf.Variable(1., dtype=tf.float64, name="length_scale")
        self.signal_amplitude = tf.Variable(1., dtype=tf.float64, name="signal_amplitude")
        self.noise_amplitude = tf.Variable(1e-1, dtype=tf.float64, name="noise_amplitude")

    def initialize_hyperparameters(self, init_minval=0.1, init_maxval=1.):
        length_scale = BoundedVariable(tf.random.uniform(shape=(1,),
                                                         minval=init_minval,
                                                         maxval=init_maxval,
                                                         dtype=tf.float64),
                                       lower=1e-4,
                                       upper=1e4,
                                       dtype=tf.float64)

        signal_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                             minval=init_minval,
                                                             maxval=init_maxval,
                                                             dtype=tf.float64),
                                           lower=1e-4,
                                           upper=1e2,
                                           dtype=tf.float64)

        noise_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                            minval=init_minval,
                                                            maxval=init_maxval,
                                                            dtype=tf.float64),
                                          lower=1e-6,
                                          upper=1e2,
                                          dtype=tf.float64)

        return length_scale, signal_amplitude, tf.cast(1e-1) #noise_amplitude

    def fit(self, xs, ys, optimizer='l-bfgs-b', optimizer_restarts=1, iters=1000, trace=False,
            err_level="catch", median_heuristic_only=False) -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        # Calculate median distance between permutations
        perm_pointwise_dists = perm_pointwise_distance(xs, xs)
        # Filter 0s
        perm_pointwise_dists = tf.gather_nd(perm_pointwise_dists, tf.where(perm_pointwise_dists != 0.))

        median_perm_dist = tfp.stats.percentile(perm_pointwise_dists, 50, axis=0)

        if median_heuristic_only:
            self.length_scale.assign(median_perm_dist)

            self.trained.assign(True)

            return

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        def negative_perm_gp_log_likelihood(length_scale, signal_amplitude, noise_amplitude):

            gp = DiscreteGaussianProcess(kernel=self.kernel_name,
                                         kernel_args=self.kernel_args,
                                         input_dim=self.input_dim,
                                         signal_amplitude=signal_amplitude,
                                         length_scales=length_scale,
                                         noise_amplitude=noise_amplitude)

            return -gp.log_pdf(xs, ys, normalize_with_input=True)

        best_loss = np.inf

        j = 0

        while j < optimizer_restarts:

            j = j + 1

            # Reinitialize hyperparameters
            hyperparams = self.initialize_hyperparameters()

            length_scale, signal_amplitude, noise_amplitude = hyperparams

            logger.info(f"Optimization round {j} / {optimizer_restarts}.")

            loss = np.inf

            try:

                loss, converged, diverged = bounded_minimize(function=lambda l, s: negative_perm_gp_log_likelihood(l, s, noise_amplitude),
                                                             vs=(length_scale, signal_amplitude),
                                                             parallel_iterations=10,
                                                             max_iterations=iters)

                if diverged:
                    logger.error(f"Model diverged, restarting iteration {j}!")
                    j = j - 1
                    continue

            except tf.errors.InvalidArgumentError as e:
                logger.error(str(e))
                j = j - 1

                if err_level == "raise":
                    raise e

                elif err_level == "catch":
                    continue

            except Exception as e:
                logger.exception("Iteration {} failed: {}".format(j + 1, str(e)))
                j -= 1

                if err_level == "raise":
                    raise e

                elif err_level == "catch":
                    continue

            if loss < best_loss:
                logger.info(f"New best objective value: {loss:.4f}")

                best_loss = loss

                # Reassign variables
                self.length_scale.assign(length_scale()[0])
                self.signal_amplitude.assign(signal_amplitude()[0])
                # self.noise_amplitude.assign(noise_amplitude()[0])

                print(self.length_scale)
                print(self.signal_amplitude)
                print(self.noise_amplitude)
            else:
                logger.info(f"Loss: {loss:.4f}")

            if np.isnan(loss) or np.isinf(loss):
                logger.error(f"Iteration {j}: Loss was {loss}, restarting training iteration!")
                j = j - 1

        self.trained.assign(True)

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
            model = model | (self.xs, self.ys)

            mean, var = model.predict(xs, latent=False)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        if numpy:
            means = means.numpy()
            variances = variances.numpy()

        return means, variances

    def log_prob(self, xs, ys, use_conditioning_data=True, latent=True, numpy=False):

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        log_prob = 0.

        for i, model in enumerate(self.models):

            cond_model = model | (self.xs, self.ys)

            if use_conditioning_data:
                model_log_prob = cond_model.log_pdf(xs,
                                                    ys,
                                                    latent=latent,
                                                    with_jitter=False,
                                                    normalize_with_training_data=True)
            else:
                # Normalize model to the regime on which the models were trained
                norm_xs = cond_model.normalize_with_training_data(xs, output=False)
                norm_ys = cond_model.normalize_with_training_data(ys, output=True)

                model_log_prob = model.log_pdf(norm_xs,
                                               norm_ys,
                                               latent=latent,
                                               with_jitter=False)

            log_prob = log_prob + model_log_prob

        if numpy:
            log_prob = log_prob.numpy()

        return log_prob

    def create_gps(self):
        self.models.clear()

        for i in range(self.output_dim):
            gp = DiscreteGaussianProcess(kernel=self.kernel_name,
                                         kernel_args=self.kernel_args,
                                         input_dim=self.input_dim,
                                         signal_amplitude=self.signal_amplitude,
                                         length_scales=self.length_scale,
                                         noise_amplitude=self.noise_amplitude)

            self.models.append(gp)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = PermutationGPModel.from_config(config)

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):
        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "distance_kind": self.distance_kind,
            "input_dim": self.input_dim,
        }

    @staticmethod
    def from_config(config):
        return PermutationGPModel(**config)

    def _validate_and_convert(self, xs, output=False):
        if isinstance(xs, (list, tuple)):
            xs = tf.stack(xs, axis=0)

        return super()._validate_and_convert(xs, output=output)
