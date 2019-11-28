import logging

import numpy as np
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

from .abstract_model_v2 import ModelError
from .gpar_v2 import GPARModel

from boa.core.utils import setup_logger
from boa.core.gp import GaussianProcess

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/mf_gpar.log")


class MatrixFactorizedGPARModel(GPARModel):
    LLLS_MAT = "left_log_length_scale_matrix"
    RLLS_MAT = "right_log_length_scale_matrix"

    def __init__(self,
                 kernel: str,
                 num_optimizer_restarts: int,
                 latent_dim: int,
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):
        super(MatrixFactorizedGPARModel, self).__init__(kernel=kernel,
                                                        num_optimizer_restarts=num_optimizer_restarts,
                                                        verbose=verbose,
                                                        name=name,
                                                        **kwargs)

        self.latent_dim = latent_dim

        self.output_length_scales = []

    def _set_data(self, xs, ys) -> tf.keras.Model:
        # Validate and assign inputs
        super(MatrixFactorizedGPARModel, self)._set_data(xs, ys)

        # Create TF variables for the hyperparameters

        # Create low rank representation for the input length scales
        self.left_length_scale_matrix = tf.Variable(
            tf.ones((self.output_dim, self.latent_dim), dtype=tf.float64),
            trainable=False,
            name=self.LLLS_MAT)

        self.right_length_scale_matrix = tf.Variable(
            tf.ones((self.latent_dim, self.input_dim), dtype=tf.float64),
            trainable=False,
            name=self.RLLS_MAT)

        # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix,
                                        self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # Length scales for the output dimensions only
            out_length_scales = tf.Variable(tf.ones(i, dtype=tf.float64),
                                            name=f"output_length_scale_dim_{i}")

            self.output_length_scales.append(out_length_scales)

            # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :],
                                       out_length_scales),
                                      axis=0)

            self.length_scales.append(length_scales)

    def create_hyperparameters(self) -> Vars:

        vs = Vars(tf.float64)

        # Create container for matrix factors
        vs.bnd(init=tf.ones((self.output_dim, self.latent_dim),
                            dtype=tf.float64),
               lower=1e-10,
               upper=1e4,
               name=self.LLLS_MAT)

        vs.bnd(init=tf.ones((self.latent_dim, self.input_dim),
                            dtype=tf.float64),
               lower=1e-10,
               upper=1e4,
               name=self.RLLS_MAT)

        # Create the rest of the hyperparameters

        for i in range(self.output_dim):
            # Note the scaling in dimension with the index
            vs.bnd(init=tf.ones(i, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=f"{i}/output_length_scales")

            # GP variance
            vs.pos(init=tf.ones(1, dtype=tf.float64),
                   name=f"{i}/signal_amplitude")

            # Noise variance: bound between 1e-4 and 1e4
            vs.bnd(init=tf.ones(1, dtype=tf.float64),
                   lower=1e-4,
                   upper=1e4,
                   name=f"{i}/noise_amplitude")

        return vs

    def initialize_hyperparameters(self, vs: Vars, length_scale_init="random"):

        if length_scale_init == "median":
            ls_init = tf.sqrt(self.xs_euclidean_percentiles[2] / self.latent_dim)

            ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                       self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

            ls_rand_range = tf.sqrt(ls_rand_range / self.latent_dim)

            llls_init = ls_init + tf.random.uniform(shape=(self.output_dim, self.latent_dim),
                                                    dtype=tf.float64,
                                                    minval=-ls_rand_range,
                                                    maxval=ls_rand_range)

            rlls_init = ls_init + tf.random.uniform(shape=(self.latent_dim, self.input_dim),
                                                    dtype=tf.float64,
                                                    minval=-ls_rand_range,
                                                    maxval=ls_rand_range)

            vs.assign(self.LLLS_MAT, llls_init)
            vs.assign(self.RLLS_MAT, rlls_init)

        else:
            vs.assign(self.LLLS_MAT,
                      tf.random.normal(shape=(self.output_dim, self.latent_dim),
                                       dtype=tf.float64,
                                       mean=tf.cast(np.log(0.5 / self.latent_dim), dtype=tf.float64),
                                       stddev=tf.cast(tf.sqrt(2 / (self.output_dim + self.latent_dim)),
                                                      dtype=tf.float64)))

            vs.assign(self.RLLS_MAT,
                      tf.random.normal(shape=(self.latent_dim, self.input_dim),
                                       dtype=tf.float64,
                                       mean=tf.cast(np.log(0.5 / self.latent_dim), dtype=tf.float64),
                                       stddev=tf.cast(tf.sqrt(2 / (self.latent_dim + self.input_dim)),
                                                      dtype=tf.float64)))

        for i in range(self.output_dim):

            if length_scale_init == "median":
                ls_init = self.ys_euclidean_percentiles[2]

                ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[2] - self.ys_euclidean_percentiles[0],
                                           self.ys_euclidean_percentiles[4] - self.ys_euclidean_percentiles[2])

                ls_init += tf.random.uniform(shape=(i,),
                                            minval=-ls_rand_range,
                                            maxval=ls_rand_range,
                                            dtype=tf.float64)

                vs.assign(f"{i}/output_length_scales", ls_init)

            else:
                vs.assign(f"{i}/output_length_scales",
                          tf.random.uniform(shape=(i,),
                                            minval=self.init_minval,
                                            maxval=self.init_maxval,
                                            dtype=tf.float64))

            vs.assign(f"{i}/signal_amplitude",
                      tf.random.uniform(shape=(1,),
                                        minval=self.init_minval,
                                        maxval=self.init_maxval,
                                        dtype=tf.float64))

            vs.assign(f"{i}/noise_amplitude",
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

        # Create dummy variables for training
        vs = self.create_hyperparameters()

        # Define i-th GP training loss
        def negative_gp_log_likelihood(idx, signal_amplitude, length_scales, noise_amplitude):

            gp = GaussianProcess(kernel=self.kernel_name,
                                 signal_amplitude=signal_amplitude,
                                 length_scales=length_scales,
                                 noise_amplitude=noise_amplitude)

            gp_input = tf.concat((self.xs, self.ys[:, :idx]), axis=1)

            return -gp.log_pdf(gp_input, self.ys[:, idx:idx + 1], normalize=True)

        # Define the GPAR training loss
        def negative_mf_gpar_log_likelihood(vs):
            losses = []

            # Create length scale matrix from its left and right factors
            length_scales = tf.matmul(vs[self.LLLS_MAT],
                                      vs[self.RLLS_MAT])

            for i in range(self.output_dim):
                # Create a single length scale vector by concatenating
                # the input and the output length scales
                gp_length_scales = tf.concat((length_scales[i, :],
                                              vs[f"{i}/output_length_scales"]),
                                             axis=0)

                losses.append(
                    negative_gp_log_likelihood(idx=i,
                                               signal_amplitude=vs[f"{i}/signal_amplitude"],
                                               length_scales=gp_length_scales,
                                               noise_amplitude=vs[f"{i}/noise_amplitude"]))

            return tf.add_n(losses)

        best_loss = np.inf

        i = 0
        # Train N MF-GPAR models and select the best one
        while i < self.num_optimizer_restarts:

            i += 1

            if self.verbose:
                print("-------------------------------")
                print(f"Training iteration {i}")
                print("-------------------------------")

            # Re-initialize to a random configuration
            self.initialize_hyperparameters(vs, length_scale_init="median")

            loss = np.inf

            try:
                # Perform L-BFGS-B optimization
                loss = minimise_l_bfgs_b(negative_mf_gpar_log_likelihood, vs)

            except Exception as e:
                print("Iteration {} failed: {}".format(i, str(e)))

            if loss < best_loss:

                if self.verbose:
                    print("New best loss: {:.3f}".format(loss))

                best_loss = loss
                self.models.clear()

                # Assign the hyperparameters to the model variables
                self.left_length_scale_matrix.assign(vs[self.LLLS_MAT])
                self.right_length_scale_matrix.assign(vs[self.RLLS_MAT])

                input_length_scales = tf.matmul(self.left_length_scale_matrix,
                                                self.right_length_scale_matrix)

                for j in range(self.output_dim):
                    self.output_length_scales[j].assign(vs[f"{j}/output_length_scales"])
                    self.gp_variances[j].assign(vs[f"{j}/signal_amplitude"])
                    self.noise_variances[j].assign(vs[f"{j}/noise_amplitude"])

                    length_scales = tf.concat((input_length_scales[j, :],
                                               self.output_length_scales[j]),
                                               axis=0)

                    gp = GaussianProcess(kernel=self.kernel_name,
                                         signal_amplitude=self.gp_variances[j],
                                         length_scales=length_scales,
                                         noise_amplitude=self.noise_variances[j])

                    self.models.append(gp)

            elif self.verbose:
                print("Loss: {:.3f}".format(loss))

            if np.isnan(loss):
                print("Loss was NaN, restarting training iteration!")
                i -= 1
