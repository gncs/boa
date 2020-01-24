import logging
import json

import numpy as np
import tensorflow as tf

from tqdm import trange

from boa.models.abstract_model import ModelError
from .gpar import GPARModel

from boa.core.utils import setup_logger
from boa.core.gp import GaussianProcess

from boa.core.variables import BoundedVariable
from boa.core.optimize import bounded_minimize

logger = setup_logger(__name__, level=logging.DEBUG, to_console=True, log_file="logs/mf_gpar.log")


class MatrixFactorizedGPARModel(GPARModel):
    LLS_MAT = "left_length_scale_matrix"
    RLS_MAT = "right_length_scale_matrix"

    def __init__(self,
                 kernel: str,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 initialization_heuristic: str = "median",
                 verbose: bool = False,
                 name="matrix_factorized_gpar_model",
                 **kwargs):

        super(MatrixFactorizedGPARModel, self).__init__(
            kernel=kernel,
            input_dim=input_dim,
            output_dim=output_dim,
            initialization_heuristic=initialization_heuristic,
            _create_length_scales=False,  # Never create the length scales in the parent class
            verbose=verbose,
            name=name,
            **kwargs)

        self.latent_dim = latent_dim

        self.output_length_scales = []

        # Create TF variables for the hyperparameters

        # Create low rank representation for the input length scales
        self.left_length_scale_matrix = tf.Variable(tf.ones((self.output_dim, self.latent_dim), dtype=tf.float64),
                                                    trainable=False,
                                                    name=self.LLS_MAT)

        self.right_length_scale_matrix = tf.Variable(tf.ones((self.latent_dim, self.input_dim), dtype=tf.float64),
                                                     trainable=False,
                                                     name=self.RLS_MAT)

        # # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # Length scales for the output dimensions only
            out_length_scales = tf.Variable(tf.ones(i, dtype=tf.float64),
                                            name=f"{i}/output_length_scale",
                                            trainable=False)

            self.output_length_scales.append(out_length_scales)

            # # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :], out_length_scales), axis=0)

            self.length_scales.append(length_scales)

    def copy(self, name=None):

        mf_gpar = super(MatrixFactorizedGPARModel, self).copy(name=name)

        input_length_scales = tf.matmul(mf_gpar.left_length_scale_matrix, mf_gpar.right_length_scale_matrix)

        for i in range(self.output_dim):
            mf_gpar.length_scales[i] = tf.concat((input_length_scales[i, :], mf_gpar.output_length_scales[i]), axis=0)

        return mf_gpar

    def initialize_hyperparameters(self, length_scale_init="random", init_minval=0.5, init_maxval=2.0):

        if length_scale_init == "median":
            ls_init = tf.sqrt(self.xs_euclidean_percentiles[2] / self.latent_dim)

            ls_rand_range = tf.minimum(self.xs_euclidean_percentiles[2] - self.xs_euclidean_percentiles[0],
                                       self.xs_euclidean_percentiles[4] - self.xs_euclidean_percentiles[2])

            ls_rand_range = tf.sqrt(ls_rand_range / self.latent_dim)

            # left length scale
            lls_init = ls_init + tf.random.uniform(
                shape=(self.output_dim, self.latent_dim), dtype=tf.float64, minval=-ls_rand_range, maxval=ls_rand_range)

            # right length scale
            rls_init = ls_init + tf.random.uniform(
                shape=(self.latent_dim, self.input_dim), dtype=tf.float64, minval=-ls_rand_range, maxval=ls_rand_range)

        else:

            lls_init = tf.random.normal(shape=(self.output_dim, self.latent_dim),
                                        dtype=tf.float64,
                                        mean=tf.cast(np.log(0.5 / self.latent_dim), dtype=tf.float64),
                                        stddev=tf.cast(tf.sqrt(2 / (self.output_dim + self.latent_dim)),
                                                       dtype=tf.float64))

            rls_init = tf.random.normal(shape=(self.latent_dim, self.input_dim),
                                        dtype=tf.float64,
                                        mean=tf.cast(np.log(0.5 / self.latent_dim), dtype=tf.float64),
                                        stddev=tf.cast(tf.sqrt(2 / (self.latent_dim + self.input_dim)),
                                                       dtype=tf.float64))

        # Create container for matrix factors:
        # Left length scale (LLS) matrix and Right length scale (RLS) matrix
        lls_mat = BoundedVariable(lls_init, lower=1e-10, upper=1e2, dtype=tf.float64)

        rls_mat = BoundedVariable(rls_init, lower=1e-10, upper=1e2, dtype=tf.float64)

        # Create the rest of the hyperparameters
        output_length_scales = []
        signal_amplitudes = []
        noise_amplitudes = []

        for i in range(self.output_dim):

            if length_scale_init == "median":
                ls_init = self.ys_euclidean_percentiles[2]

                ls_rand_range = tf.minimum(self.ys_euclidean_percentiles[2] - self.ys_euclidean_percentiles[0],
                                           self.ys_euclidean_percentiles[4] - self.ys_euclidean_percentiles[2])

                ls_init += tf.random.uniform(shape=(i, ), minval=-ls_rand_range, maxval=ls_rand_range, dtype=tf.float64)
            else:
                ls_init = tf.random.uniform(shape=(i, ), minval=init_minval, maxval=init_maxval, dtype=tf.float64)

            output_length_scales.append(BoundedVariable(ls_init, lower=1e-3, upper=1e2, dtype=tf.float64))

            signal_amplitudes.append(
                BoundedVariable(tf.random.uniform(shape=(1, ), minval=init_minval, maxval=init_maxval,
                                                  dtype=tf.float64),
                                lower=1e-4,
                                upper=1e4,
                                dtype=tf.float64))

            noise_amplitudes.append(
                BoundedVariable(tf.random.uniform(shape=(1, ), minval=init_minval, maxval=init_maxval,
                                                  dtype=tf.float64),
                                lower=1e-6,
                                upper=1e2,
                                dtype=tf.float64))

        return lls_mat, rls_mat, output_length_scales, signal_amplitudes, noise_amplitudes

    def fit(self,
            xs,
            ys,
            optimizer="adam",
            optimizer_restarts=1,
            trace=True,
            iters=200,
            rate=1e-2,
            tolerance=1e-5,
            err_level="catch") -> None:

        xs, ys = self._validate_and_convert_input_output(xs, ys)

        logger.info(f"Training data supplied with xs shape {xs.shape} and ys shape {ys.shape}, training!")

        self._calculate_statistics_for_median_initialization_heuristic(xs, ys)

        # Define i-th GP training loss
        def negative_gp_log_likelihood(idx, signal_amplitude, length_scales, noise_amplitude):

            gp = GaussianProcess(kernel=self.kernel_name,
                                 signal_amplitude=signal_amplitude,
                                 length_scales=length_scales,
                                 noise_amplitude=noise_amplitude)

            gp_input = tf.concat((xs, ys[:, :idx]), axis=1)

            return -gp.log_pdf(gp_input, ys[:, idx:idx + 1], normalize=True)

        # Define the GPAR training loss
        def negative_mf_gpar_log_likelihood(lls_mat, rls_mat, output_length_scales, signal_amplitudes,
                                            noise_amplitudes):
            losses = []

            # Create length scale matrix from its left and right factors
            length_scales = tf.matmul(lls_mat, rls_mat)

            for k in range(self.output_dim):
                # Create a single length scale vector by concatenating
                # the input and the output length scales
                gp_length_scales = tf.concat((length_scales[k, :], output_length_scales[k]), axis=0)

                losses.append(
                    negative_gp_log_likelihood(idx=k,
                                               signal_amplitude=signal_amplitudes[k],
                                               length_scales=gp_length_scales,
                                               noise_amplitude=noise_amplitudes[k]))

            return tf.add_n(losses)

        best_loss = np.inf

        i = 0
        # Train N MF-GPAR models and select the best one
        while i < optimizer_restarts:

            i += 1

            logger.info("-------------------------------")
            logger.info(f"Training iteration {i}")
            logger.info("-------------------------------")

            # Re-initialize to a random configuration
            hyperparams = self.initialize_hyperparameters(length_scale_init="median")

            lls_mat, rls_mat, output_length_scales, signal_amplitudes, noise_amplitudes = hyperparams

            loss = np.inf

            try:
                if optimizer == "l-bfgs-b":

                    # Perform L-BFGS-B optimization
                    loss, converged, diverged = bounded_minimize(
                        function=negative_mf_gpar_log_likelihood,
                        vs=hyperparams,
                        parallel_iterations=10,
                        max_iterations=iters,
                        trace=trace)

                    if diverged:
                        logger.error(f"Model diverged, restarting iteration {i}!")
                        i -= 1
                        continue

                elif optimizer == "adam":
                    # Get the list of reparametrizations for the hyperparameters
                    reparams = BoundedVariable.get_reparametrizations(hyperparams, flatten=True)

                    optimizer = tf.optimizers.Adam(rate, epsilon=1e-8)

                    prev_loss = np.inf

                    with trange(iters) as t:
                        for iteration in t:
                            with tf.GradientTape(watch_accessed_variables=False) as tape:
                                tape.watch(reparams)

                                loss = negative_mf_gpar_log_likelihood(*BoundedVariable.get_all(hyperparams))

                            if tf.abs(prev_loss - loss) < tolerance:
                                logger.info(f"Loss decreased less than {tolerance}, "
                                            f"optimisation terminated at iteration {iteration}.")
                                break

                            prev_loss = loss

                            gradients = tape.gradient(loss, reparams)
                            optimizer.apply_gradients(zip(gradients, reparams))

                            t.set_description(f"Loss at iteration {iteration}: {loss:.3f}.")

            except tf.errors.InvalidArgumentError as e:
                logger.error(str(e))
                i -= 1

                if err_level == "raise":
                    raise e

                elif err_level == "catch":
                    continue

            except Exception as e:
                logger.error("Iteration {} failed: {}".format(i, str(e)))
                i -= 1

                if err_level == "raise":
                    raise e

                elif err_level == "catch":
                    continue

            if loss < best_loss:

                logger.info("New best loss: {:.3f}".format(loss))

                best_loss = loss
                self.models.clear()

                # Assign the hyperparameters to the model variables
                self.left_length_scale_matrix.assign(lls_mat())
                self.right_length_scale_matrix.assign(rls_mat())

                input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

                for j in range(self.output_dim):
                    self.output_length_scales[j].assign(output_length_scales[j]())
                    self.signal_amplitudes[j].assign(signal_amplitudes[j]())
                    self.noise_amplitudes[j].assign(noise_amplitudes[j]())

                    self.length_scales[j] = tf.concat((input_length_scales[j, :], self.output_length_scales[j]), axis=0)

            else:
                logger.info("Loss: {:.3f}".format(loss))

            if np.isnan(loss):
                logger.error("Loss was NaN, restarting training iteration!")
                i -= 1

        self.trained.assign(True)

    def create_gps(self):
        self.models.clear()

        self.length_scales.clear()

        # dimensions O x I
        input_length_scales = tf.matmul(self.left_length_scale_matrix, self.right_length_scale_matrix)

        for i in range(self.output_dim):
            # i-th length scales
            length_scales = tf.concat((input_length_scales[i, :], self.output_length_scales[i]), axis=0)

            self.length_scales.append(length_scales)

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

        model = GPARModel.from_config(config)

        model.load_weights(save_path)
        model.create_gps()

        return model

    def get_config(self):

        return {
            "name": self.name,
            "kernel": self.kernel_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "latent_dim": self.latent_dim,
            "denoising": self.denoising,
            "initialization_heuristic": self.initialization_heuristic,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_config(config, restore_num_data_points=False):
        return MatrixFactorizedGPARModel(**config)
