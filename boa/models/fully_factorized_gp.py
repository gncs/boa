import logging

import json

from typing import List

import tensorflow as tf
import numpy as np

from boa.core.gp import GaussianProcess
from boa.core.utils import setup_logger
from .abstract_model import AbstractModel, ModelError

from boa.core.variables import BoundedVariable
from boa.core.optimize import bounded_minimize

__all__ = ["FullyFactorizedGPModel"]

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/ff_gp.log")


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

        self.length_scales: List[tf.Variable] = []
        self.signal_amplitudes: List[tf.Variable] = []
        self.noise_amplitudes: List[tf.Variable] = []

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            self.length_scales.append(
                tf.Variable(tf.ones(self.input_dim, dtype=tf.float64), name=f"{i}/length_scales", trainable=False))

            self.signal_amplitudes.append(
                tf.Variable((1.0,), dtype=tf.float64, name=f"{i}/signal_amplitude", trainable=False))

            self.noise_amplitudes.append(
                tf.Variable((1.0,), dtype=tf.float64, name=f"{i}/noise_amplitude", trainable=False))

    def initialize_hyperparameters(self, length_scale_init, init_minval=0.1, init_maxval=1.0):

        if length_scale_init == "median":

            # Center on the median
            ls_init = self.xs_euclidean_percentiles[2]

            ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                       self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

            ls_init += tf.random.uniform(shape=(self.input_dim,),
                                         minval=-ls_rand_range,
                                         maxval=ls_rand_range,
                                         dtype=tf.float64)

        else:
            ls_init = tf.random.uniform(shape=(self.input_dim,),
                                        minval=init_minval,
                                        maxval=init_maxval,
                                        dtype=tf.float64)

        length_scales = BoundedVariable(ls_init, lower=1e-3, upper=1e2, dtype=tf.float64)

        signal_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                             minval=init_minval,
                                                             maxval=init_maxval,
                                                             dtype=tf.float64),
                                           lower=1e-4,
                                           upper=1e4,
                                           dtype=tf.float64)

        noise_amplitude = BoundedVariable(tf.random.uniform(shape=(1,),
                                                            minval=init_minval,
                                                            maxval=init_maxval,
                                                            dtype=tf.float64),
                                          lower=1e-6,
                                          upper=1e4)

        return length_scales, signal_amplitude, noise_amplitude

    def fit(self, xs, ys, optimizer="l-bfgs-b", optimizer_restarts=1, iters=1000, trace=False,
            err_level="catch") -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        # Optimize each dimension individually
        for i in range(self.output_dim):

            best_loss = np.inf

            # Training objective
            def negative_gp_log_likelihood(length_scales, signal_amplitude, noise_amplitude):

                gp = GaussianProcess(kernel=self.kernel_name,
                                     input_dim=self.input_dim,
                                     signal_amplitude=signal_amplitude,
                                     length_scales=length_scales,
                                     noise_amplitude=noise_amplitude)

                return -gp.log_pdf(xs, ys[:, i:i + 1], normalize_with_input=True)

            # Robust optimization
            j = 0
            while j < optimizer_restarts:

                j += 1

                # Reinitialize parameters
                hyperparams = self.initialize_hyperparameters(length_scale_init=self.initialization_heuristic)

                length_scales, signal_amplitude, noise_amplitude = hyperparams

                logger.info(f"Dimension {i} Optimization round: {j} / {optimizer_restarts}")

                loss = np.inf
                try:
                    # Perform L-BFGS-B optimization
                    loss, converged, diverged = bounded_minimize(function=negative_gp_log_likelihood,
                                                                 vs=hyperparams,
                                                                 parallel_iterations=10,
                                                                 max_iterations=iters)

                    if diverged:
                        logger.error(f"Model diverged, restarting iteration {j}! (loss was {loss:.3f})")
                        j -= 1
                        continue

                except tf.errors.InvalidArgumentError as e:
                    logger.error(str(e))
                    j -= 1

                    if err_level == "raise":
                        raise e

                    elif err_level == "catch":
                        continue

                except Exception as e:
                    logger.exception("Iteration {} failed: {}".format(i + 1, str(e)))
                    j -= 1

                    if err_level == "raise":
                        raise e

                    elif err_level == "catch":
                        continue

                if loss < best_loss:
                    logger.info(f"New best objective value for dimension {i}: {loss:.4f}")

                    best_loss = loss

                    # Reassign variables
                    self.length_scales[i].assign(length_scales())
                    self.signal_amplitudes[i].assign(signal_amplitude())
                    self.noise_amplitudes[i].assign(noise_amplitude())
                else:
                    logger.info(f"Loss for dimension {i}: {loss:.4f}")

                if np.isnan(loss) or np.isinf(loss):
                    logger.error(f"Output {i}, Iteration {j}: Loss was {loss}, restarting training iteration!")
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

    def create_gps(self):
        self.models.clear()

        for i in range(self.output_dim):
            gp = GaussianProcess(kernel=self.kernel_name,
                                 input_dim=self.input_dim,
                                 signal_amplitude=self.signal_amplitudes[i],
                                 length_scales=self.length_scales[i],
                                 noise_amplitude=self.noise_amplitudes[i])

            self.models.append(gp)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = FullyFactorizedGPModel.from_config(config)

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
    def from_config(config):
        return FullyFactorizedGPModel(**config)
