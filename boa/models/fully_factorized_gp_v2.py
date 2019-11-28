import logging

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
                 num_optimizer_restarts: int,
                 parallel: bool = False,
                 name: str = "gp_model",
                 verbose=True,
                 **kwargs):
        """
        Constructor of GP model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        :param parallel: run optimizations in parallel
        :param name: name of the GP model
        """

        super(FullyFactorizedGPModel, self).__init__(kernel=kernel,
                                                     num_optimizer_restarts=num_optimizer_restarts,
                                                     parallel=parallel,
                                                     verbose=verbose,
                                                     name=name,
                                                     **kwargs)

        self.length_scales: List[tf.Variable] = []
        self.gp_variances: List[tf.Variable] = []
        self.noise_variances: List[tf.Variable] = []

    def _set_data(self, xs, ys) -> None:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        # Validate inputs and set data
        super(FullyFactorizedGPModel, self)._set_data(xs, ys)

        # Create GP hyperparameter variables
        for i in range(self.output_dim):
            self.length_scales.append(tf.Variable(tf.ones(self.input_dim,
                                                          dtype=tf.float64),
                                                  name=f"{i}/length_scales",
                                                  trainable=False))

            self.gp_variances.append(tf.Variable((1.0,),
                                                 dtype=tf.float64,
                                                 name=f"{i}/signal_amplitude",
                                                 trainable=False))

            self.noise_variances.append(tf.Variable((1.0,),
                                                    dtype=tf.float64,
                                                    name=f"{i}/noise_amplitude",
                                                    trainable=False))

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

    def initialize_hyperparameters(self, vs: Vars, index, length_scale_init="random") -> None:

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
                                        minval=self.init_minval,
                                        maxval=self.init_maxval,
                                        dtype=tf.float64)
        vs.assign(ls_name, ls_init)

        vs.assign(gp_var_name, tf.random.uniform(shape=(1,),
                                                 minval=self.init_minval,
                                                 maxval=self.init_maxval,
                                                 dtype=tf.float64))

        vs.assign(noise_var_name, tf.random.uniform(shape=(1,),
                                                    minval=self.init_minval,
                                                    maxval=self.init_maxval,
                                                    dtype=tf.float64))

    def fit(self, xs=None, ys=None, init_minval=0.5, init_maxval=2.0) -> None:

        if xs is None and ys is None:
            logger.info("No training data supplied, retraining using data already present!")

        elif xs is not None and ys is not None:
            logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")
            self._set_data(xs, ys)

        else:
            message = f"Training conditions inconsistent, xs were {type(xs)} and ys were {type(xs)}!"

            logger.error(message)
            raise ModelError(message)

        self.models.clear()

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

                return -gp.log_pdf(self.xs, self.ys[:, i:i + 1], normalize=True)

            # Robust optimization
            for j in range(self.num_optimizer_restarts):

                # Reinitialize parameters
                self.initialize_hyperparameters(vs,
                                                index=i,
                                                length_scale_init="dim_median")

                logger.info("Optimization round: {} / {}".format(j + 1, self.num_optimizer_restarts))

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
                    self.gp_variances[i].assign(vs[gp_var_name])
                    self.noise_variances[i].assign(vs[noise_var_name])

            best_gp = GaussianProcess(kernel=self.kernel_name,
                                      signal_amplitude=self.gp_variances[i],
                                      length_scales=self.length_scales[i],
                                      noise_amplitude=self.noise_variances[i])
            self.models.append(best_gp)

    def predict_batch(self, xs):

        if len(self.models) == 0:
            raise ModelError("The model has not been trained yet!")

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

        return means, variances
