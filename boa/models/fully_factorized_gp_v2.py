import logging

import json

from typing import Tuple, List

import tensorflow as tf
from varz.tensorflow import minimise_l_bfgs_b, Vars

import numpy as np

from boa.core.gp import GaussianProcess
from boa.core.utils import setup_logger
from .abstract_model_v2 import AbstractModel, ModelError

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
                 _num_starting_data_points: int = 0,
                 **kwargs):


        super(FullyFactorizedGPModel, self).__init__(kernel=kernel,
                                                     input_dim=input_dim,
                                                     output_dim=output_dim,
                                                     parallel=parallel,
                                                     verbose=verbose,
                                                     _num_starting_data_points=_num_starting_data_points,
                                                     name=name,
                                                     **kwargs)

        self.initialization_heuristic = initialization_heuristic

        self.length_scales: List[tf.Variable] = []
        self.signal_amplitudes: List[tf.Variable] = []
        self.noise_amplitudes: List[tf.Variable] = []

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            self.length_scales.append(tf.Variable(tf.ones(self.input_dim,
                                                          dtype=tf.float64),
                                                  name=f"{i}/length_scales",
                                                  trainable=False))

            self.signal_amplitudes.append(tf.Variable((1.0,),
                                                      dtype=tf.float64,
                                                      name=f"{i}/signal_amplitude",
                                                      trainable=False))

            self.noise_amplitudes.append(tf.Variable((1.0,),
                                                     dtype=tf.float64,
                                                     name=f"{i}/noise_amplitude",
                                                     trainable=False))

    def copy(self, name=None):
        ff_gp = FullyFactorizedGPModel(kernel=self.kernel_name,
                                       input_dim=self.input_dim,
                                       output_dim=self.output_dim,
                                       initialization_heuristic=self.initialization_heuristic,
                                       parallel=self.parallel,
                                       name=name if name is not None else self.name)

        # Copy stored hyperparameter values
        for i in range(self.output_dim):
            ff_gp.length_scales[i].assign(self.length_scales[i])
            ff_gp.signal_amplitudes[i].assign(self.signal_amplitudes[i])
            ff_gp.noise_amplitudes[i].assign(self.noise_amplitudes[i])

        # Copy data
        ff_gp.xs = tf.Variable(self.xs, name=self.XS_NAME, trainable=False)
        ff_gp.ys = tf.Variable(self.ys, name=self.YS_NAME, trainable=False)

        # Copy miscellaneous stuff
        ff_gp.trained.assign(self.trained)

        return ff_gp

    def create_hyperparameters(self) -> Vars:

        vs = Vars(tf.float64)

        for i in range(self.output_dim):
            ls_name = f"{i}/length_scales"
            gp_var_name = f"{i}/signal_amplitude"
            noise_var_name = f"{i}/noise_amplitude"

            # Length scales
            vs.bnd(init=tf.ones(self.input_dim, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=ls_name)

            # GP variance
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=gp_var_name)

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=noise_var_name)

        return vs

    def initialize_hyperparameters(self,
                                   vs: Vars,
                                   index,
                                   length_scale_init="random",
                                   init_minval=0.5,
                                   init_maxval=2.0) -> None:

        ls_name = f"{index}/length_scales"
        gp_var_name = f"{index}/signal_amplitude"
        noise_var_name = f"{index}/noise_amplitude"

        if length_scale_init == "median":

            # Center on the median
            ls_init = self.xs_euclidean_percentiles[2]

            ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                       self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

            ls_init += tf.random.uniform(shape=(self.input_dim,),
                                         minval=-ls_rand_range,
                                         maxval=ls_rand_range,
                                         dtype=tf.float64)

        elif length_scale_init == "dim_median":

            ls_init = self.xs_per_dim_percentiles[:, 2]

            ls_rand_range = tf.minimum(self.xs_per_dim_percentiles[:, 2] - self.xs_per_dim_percentiles[:, 0],
                                       self.xs_per_dim_percentiles[:, 4] - self.xs_per_dim_percentiles[:, 2])

            tf.random.uniform(shape=(self.input_dim,),
                              minval=-ls_rand_range,
                              maxval=ls_rand_range,
                              dtype=tf.float64)

        else:
            ls_init = tf.random.uniform(shape=(self.input_dim,),
                                        minval=init_minval,
                                        maxval=init_maxval,
                                        dtype=tf.float64)
        vs.assign(ls_name, ls_init)

        vs.assign(gp_var_name, tf.random.uniform(shape=(1,),
                                                 minval=init_minval,
                                                 maxval=init_maxval,
                                                 dtype=tf.float64))

        vs.assign(noise_var_name, tf.random.uniform(shape=(1,),
                                                    minval=init_minval,
                                                    maxval=init_maxval,
                                                    dtype=tf.float64))

    def fit(self, xs, ys, optimizer_restarts=1) -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        vs = self.create_hyperparameters()

        # Optimize each dimension individually
        for i in range(self.output_dim):
            ls_name = f"{i}/length_scales"
            gp_var_name = f"{i}/signal_amplitude"
            noise_var_name = f"{i}/noise_amplitude"

            best_loss = np.inf

            # Training objective
            def negative_gp_log_likelihood(signal_amplitude, length_scales, noise_amplitude):

                gp = GaussianProcess(kernel=self.kernel_name,
                                     signal_amplitude=signal_amplitude,
                                     length_scales=length_scales,
                                     noise_amplitude=noise_amplitude)

                return -gp.log_pdf(xs, ys[:, i:i + 1], normalize=True)

            # Robust optimization
            for j in range(optimizer_restarts):

                # Reinitialize parameters
                self.initialize_hyperparameters(vs,
                                                index=i,
                                                length_scale_init=self.initialization_heuristic)

                logger.info("Optimization round: {} / {}".format(j + 1, optimizer_restarts))

                loss = np.inf
                try:
                    # Perform L-BFGS-B optimization
                    loss = minimise_l_bfgs_b(lambda v: negative_gp_log_likelihood(v[gp_var_name],
                                                                                  v[ls_name],
                                                                                  v[noise_var_name]),
                                             vs,
                                             names=[ls_name,
                                                    gp_var_name,
                                                    noise_var_name])
                except Exception as e:
                    logger.exception("Iteration {} failed: {}".format(i + 1, str(e)))
                    raise e

                if loss < best_loss:
                    logger.info(f"New best objective value for dimension {i}: {loss:.4f}")

                    best_loss = loss

                    # Reassign variables
                    self.length_scales[i].assign(vs[ls_name])
                    self.signal_amplitudes[i].assign(vs[gp_var_name])
                    self.noise_amplitudes[i].assign(vs[noise_var_name])

        self.trained.assign(True)

    def predict(self, xs, numpy=False):

        if not self.trained:
            logger.warning("Using untrained model for prediction!")

        if len(self.models) < self.output_dim:
            logger.info("GPs haven't been cached yet, creating them now.")
            self.create_gps()

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        if xs.shape[1] != self.input_dim:
            raise ModelError("xs with shape {} must have 1st dimension (0 indexed) {}!".format(xs.shape,
                                                                                               self.input_dim))

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

    def create_gps(self):
        self.models.clear()

        for i in range(self.output_dim):
            gp = GaussianProcess(kernel=self.kernel_name,
                                 signal_amplitude=self.signal_amplitudes[i],
                                 length_scales=self.length_scales[i],
                                 noise_amplitude=self.noise_amplitudes[i])

            self.models.append(gp)

    @staticmethod
    def restore(save_path):

        with open(save_path + ".json", "r") as config_file:
            config = json.load(config_file)

        model = FullyFactorizedGPModel.from_config(config, restore_num_data_points=True)

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
            "num_data_points": self.xs.shape[0]
        }

    @staticmethod
    def from_config(config, restore_num_data_points=False):

        return FullyFactorizedGPModel(kernel=config["kernel"],
                                      input_dim=config["input_dim"],
                                      output_dim=config["output_dim"],
                                      initialization_heuristic=config["initialization_heuristic"],
                                      parallel=config["parallel"],
                                      verbose=config["verbose"],
                                      _num_starting_data_points=config["num_data_points"] if restore_num_data_points else 0)


