import logging
import os

from typing import Tuple, List

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import AbstractModel, ModelError
from boa.core import GaussianProcess, PermutationVariable, setup_logger

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/gpar.log")


class GPARModel(AbstractModel):

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 learn_permutation: bool = False,
                 denoising: bool = True,
                 verbose: bool = False,
                 name: str = "gpar_model",
                 **kwargs):
        """
        Constructor of GPAR model.

        :param kernel: name of kernel
        :param num_optimizer_restarts: number of times the optimization of the hyperparameters is restarted
        :param verbose: log optimization of hyperparameters
        """

        super(GPARModel, self).__init__(kernel=kernel,
                                        num_optimizer_restarts=num_optimizer_restarts,
                                        verbose=verbose,
                                        name=name,
                                        **kwargs)

        self.denoising = denoising

        self.length_scales: List[tf.Variable] = []
        self.gp_variances: List[tf.Variable] = []
        self.noise_variances: List[tf.Variable] = []

    def _set_data(self, xs, ys) -> None:
        """
        This override of | is implements the inference of the posterior GP model from the inputs (xs, ys)

        :param inputs: tuple of array-like xs and ys, i.e. input points and their corresponding function values
        :return: None
        """

        # Validate inputs
        super(GPARModel, self)._set_data(xs, ys)

        self.output_perm = tf.eye(self.output_dim, dtype=tf.float64)

        # Create TF variables for each of the hyperparameters, so that
        # we can use Keras' serialization features
        for i in range(self.output_dim):
            # Note the scaling in dimension
            self.length_scales.append(tf.Variable(tf.ones(self.input_dim + i,
                                                          dtype=tf.float64),
                                                  name=f"{i}/length_scales",
                                                  trainable=False))

            self.gp_variances.append(tf.Variable((1,),
                                                 dtype=tf.float64,
                                                 name=f"{i}/signal_amplitude",
                                                 trainable=False))

            self.noise_variances.append(tf.Variable((1,),
                                                    dtype=tf.float64,
                                                    name=f"{i}/noise_amplitude",
                                                    trainable=False))

    def create_hyperparameters(self) -> Vars:
        """
        Creates the hyperparameter container that the model uses
        and creates the constrained hyperparameter variables in it,
        initialized to some dummy values.

        *Note*: It is not safe to use the initialized values for training,
        always call initialize_hyperparameters first!

        :return: Varz variable container
        """

        vs = Vars(tf.float64)

        for i in range(self.output_dim):
            ls_name = f"{i}/length_scales"
            gp_var_name = f"{i}/signal_amplitude"
            noise_var_name = f"{i}/noise_amplitude"

            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(self.input_dim + i, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=ls_name)

            # GP variance
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e3,
                   name=gp_var_name)

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-6,
                   upper=1e3,
                   name=noise_var_name)

        return vs

    def initialize_hyperparameters(self, vs, index, length_scale_init="random") -> None:
        # for i in range(self.output_dim):

        i = index

        ls_name = f"{i}/length_scales"
        gp_var_name = f"{i}/signal_amplitude"
        noise_var_name = f"{i}/noise_amplitude"

        if length_scale_init == "median":

            # Center on the medians, treat the inputs and the outputs separately
            xs_ls_init = self.xs_euclidean_percentiles[2]
            ys_ls_init = self.ys_euclidean_percentiles[2]

            xs_ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                          self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

            ys_ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[2] - self.ys_euclidean_percentiles[0],
                                          self.ys_euclidean_percentiles[4] - self.ys_euclidean_percentiles[2])

            xs_ls_init += tf.random.uniform(shape=(self.input_dim,),
                                            minval=-xs_ls_rand_range,
                                            maxval=xs_ls_rand_range,
                                            dtype=tf.float64)

            ys_ls_init += tf.random.uniform(shape=(i,),
                                            minval=-ys_ls_rand_range,
                                            maxval=ys_ls_rand_range,
                                            dtype=tf.float64)

            # Once the inputs and outputs have been initialized separately, concatenate them
            ls_init = tf.concat((xs_ls_init, ys_ls_init), axis=0)

        elif length_scale_init == "dim_median":
            xs_ls_init = self.xs_per_dim_percentiles[:, 2]
            xs_ls_rand_range = tf.minimum(self.xs_per_dim_percentiles[:, 2] - self.xs_per_dim_percentiles[:, 0],
                                          self.xs_per_dim_percentiles[:, 4] - self.xs_per_dim_percentiles[:, 2])

            xs_ls_init += tf.random.uniform(shape=(self.input_dim,),
                                            minval=-xs_ls_rand_range,
                                            maxval=xs_ls_rand_range,
                                            dtype=tf.float64)

            ys_ls_init = self.ys_per_dim_percentiles[:, 2]
            ys_ls_rand_range = tf.minimum(self.ys_per_dim_percentiles[:, 2] - self.ys_per_dim_percentiles[:, 0],
                                          self.ys_per_dim_percentiles[:, 4] - self.ys_per_dim_percentiles[:, 2])

            ys_ls_init += tf.random.uniform(shape=(i,),
                                            minval=-ys_ls_rand_range,
                                            maxval=ys_ls_rand_range,
                                            dtype=tf.float64)

            # Once the inputs and outputs have been initialized separately, concatenate them
            ls_init = tf.concat((xs_ls_init, ys_ls_init), axis=0)

        else:
            ls_init = tf.random.uniform(shape=(self.input_dim + i,),
                                        minval=self.init_minval,
                                        maxval=self.init_maxval,
                                        dtype=tf.float64)
        vs.assign(ls_name, ls_init)

        vs.assign(gp_var_name,
                  tf.random.uniform(shape=(1,),
                                    minval=self.init_minval,
                                    maxval=self.init_maxval,
                                    dtype=tf.float64))

        vs.assign(noise_var_name,
                  tf.random.uniform(shape=(1,),
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

        # Optimize dimensions individually
        for i in range(self.output_dim):
            length_scales_name = f"{i}/length_scales"
            sig_amp_name = f"{i}/signal_amplitude"
            noise_amp = f"{i}/noise_amplitude"

            best_loss = np.inf

            # Define i-th GP training loss
            def negative_gp_log_likelihood(signal_amplitude, length_scales, noise_amplitude):

                gp = GaussianProcess(kernel=self.kernel_name,
                                     signal_amplitude=signal_amplitude,
                                     length_scales=length_scales,
                                     noise_amplitude=noise_amplitude)

                gp_input = tf.concat((self.xs, self.ys[:, :i]), axis=1)

                return -gp.log_pdf(gp_input, self.ys[:, i:i + 1], normalize=True)

            # Robust optimization
            j = 0

            while j < self.num_optimizer_restarts:
                j += 1

                self.initialize_hyperparameters(vs, index=i, length_scale_init="dim_median")

                loss = np.inf

                try:
                    # Perform L-BFGS-B optimization
                    loss = minimise_l_bfgs_b(lambda v: negative_gp_log_likelihood(signal_amplitude=v[sig_amp_name],
                                                                                  length_scales=v[length_scales_name],
                                                                                  noise_amplitude=v[noise_amp]),
                                             vs,
                                             names=[sig_amp_name,
                                                    length_scales_name,
                                                    noise_amp],
                                             trace=False,
                                             err_level="raise")

                except Exception as e:

                    logger.error(f"Saving: {vs[sig_amp_name]}, --- {vs[noise_amp]}")

                    if not os.path.exists(os.path.dirname("logs/" + length_scales_name)):
                        os.makedirs(os.path.dirname("logs/" + length_scales_name))

                    np.save("logs/" + length_scales_name, vs[length_scales_name].numpy())
                    np.save("logs/" + sig_amp_name, vs[sig_amp_name].numpy())
                    np.save("logs/" + noise_amp, vs[noise_amp].numpy())

                    logger.error("Iteration {} failed: {}".format(i, str(e)))

                    j = j - 1
                    continue

                if loss < best_loss:

                    logger.info(f"Output {i}, Iteration {j}: New best loss: {loss:.3f}")

                    best_loss = loss

                    # Assign the hyperparameters for each input to the model variables
                    self.length_scales[i].assign(vs[length_scales_name])
                    self.gp_variances[i].assign(vs[sig_amp_name])
                    self.noise_variances[i].assign(vs[noise_amp])

                else:
                    logger.info(f"Output {i}, Iteration {j}: Loss: {loss:.3f}")

                if np.isnan(loss) or np.isinf(loss):
                    logger.error(f"Output {i}, Iteration {j}: Loss was {loss}, restarting training iteration!")
                    j = j - 1

            best_gp = GaussianProcess(kernel=self.kernel_name,
                                      signal_amplitude=self.gp_variances[i],
                                      length_scales=self.length_scales[i],
                                      noise_amplitude=self.noise_variances[i])

            self.models.append(best_gp)

    def predict_batch(self, xs) -> Tuple[tf.Tensor, tf.Tensor]:

        if len(self.models) == 0:
            raise ModelError("The model has not been trained yet!")

        xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        if xs.shape[1] != self.input_dim:
            raise ModelError("xs with shape {} must have 1st dimension (0 indexed) {}!".format(xs.shape,
                                                                                               self.input_dim))

        means = []
        variances = []

        for i, model in enumerate(self.models):
            gp_input = tf.concat([xs] + means, axis=1)
            gp_train_input = tf.concat([self.xs, self.ys[:, :i]], axis=1)

            model = model | (gp_train_input, self.ys[:, i: i + 1])

            mean, var = model.predict(gp_input, latent=False)

            means.append(mean)
            variances.append(var)

        means = tf.concat(means, axis=1)
        variances = tf.concat(variances, axis=1)

        return means, variances
